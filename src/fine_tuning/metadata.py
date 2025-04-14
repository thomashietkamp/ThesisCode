import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import warnings

# Suppress UserWarnings from TRL aboutpacking (can be noisy)
warnings.filterwarnings("ignore", category=UserWarning,
                        module='trl.trainer.sft_trainer')
# Suppress specific BitsAndBytes warnings if needed, but be aware of them
# warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")


# --- Configuration ---
MODEL_CONFIG = {
    # Model from Hugging Face Hub - Gemma 2 9B IT is a good starting point
    # Other options: "google/gemma-7b-it", "google/gemma-2b-it"
    "base_model_name": "google/gemma-3-4b-it",
    # WARNING: Inputs longer than this WILL BE TRUNCATED.
    # If your docs are longer and truncation is not acceptable,
    # you need a different strategy (filtering, chunking, long-context model).
    "max_seq_length": 128000,
    # QLoRA bits: 4 or 8. 4 is more memory efficient.
    "quantization_bits": 8,
}

DATA_CONFIG = {
    "train_file": "data/jsonl/training.jsonl",
    # Optional: for evaluation during training
    "test_file": "data/jsonl/test.jsonl",
    "prompt_field": "prompt",
    "completion_field": "completion",
    "test_size": 0.05  # Fraction of train data to use for eval if test_file is None
}

QLORA_CONFIG = {
    # LoRA attention dimension
    "lora_r": 16,  # Standard value, can be tuned (e.g., 32, 64)
    # Alpha parameter for LoRA scaling
    "lora_alpha": 32,  # Often 2*lora_r
    # Dropout probability for LoRA layers
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    # Target modules vary by model. Use tools or examples to find them.
    # For Gemma, typical targets include query, key, value, and output layers
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

TRAINING_CONFIG = {
    # Define a unique name for your fine-tuned model
    "output_dir": "./gemma_contract_finetuned",
    # Number of training epochs
    "num_train_epochs": 1,  # Start with 1-3 epochs for SFT
    # Batch size per GPU for training
    # Adjust based on VRAM (1 or 2 often needed for large models/seq len)
    "per_device_train_batch_size": 1,
    # Batch size per GPU for evaluation
    "per_device_eval_batch_size": 1,
    # Number of steps between logging updates
    "logging_steps": 25,
    # Number of steps between saving checkpoints
    "save_steps": 100,
    # Strategy to evaluate the model performance
    "evaluation_strategy": "steps" if os.path.exists(DATA_CONFIG["test_file"]) else "no",
    "eval_steps": 100 if os.path.exists(DATA_CONFIG["test_file"]) else None,
    # Save only the best model based on evaluation loss (requires evaluation_strategy)
    "save_total_limit": 2,  # Limit the number of checkpoints saved
    "load_best_model_at_end": True if os.path.exists(DATA_CONFIG["test_file"]) else False,
    # Optimizer to use
    "optim": "paged_adamw_32bit",  # Paged optimizer for memory efficiency
    # Learning rate
    # A lower LR is often better for QLoRA (e.g., 2e-5, 1e-5)
    "learning_rate": 2e-5,
    # Weight decay for regularization
    "weight_decay": 0.001,
    # Gradient accumulation steps
    # Increase effective batch size (train_batch_size * grad_accum_steps)
    "gradient_accumulation_steps": 4,
    # Maximum gradient normal (gradient clipping)
    "max_grad_norm": 0.3,
    # Learning rate scheduler type
    "lr_scheduler_type": "cosine",  # "linear" or "cosine"
    # Number of warmup steps for the learning rate scheduler
    "warmup_ratio": 0.03,
    # Use mixed precision training
    # Set True for fp16 (faster, less memory on compatible GPUs)
    "fp16": False,
    # Set True for bf16 (requires Ampere GPU or newer) - often preferred over fp16
    "bf16": True,
    # Group texts into batches of max_seq_length for efficiency (can sometimes cause issues)
    "group_by_length": False,  # Set True if sequences vary greatly in length
    # Report results to W&B (optional)
    "report_to": "none",  # Change to "wandb" if you have it configured
}

# --- Helper Function for Formatting ---

# Adapts the prompt+completion into the format Gemma expects
# See: https://huggingface.co/google/gemma-2-9b-it#chat-template
# Using a simpler instruction format here for SFT


def format_prompt(example):
    prompt_text = example[DATA_CONFIG['prompt_field']]
    completion_text = example[DATA_CONFIG['completion_field']]
    question_text = "Please extract the document name, parties, agreement date, effective date, expiration date, renewal term, notice period to terminate renewal, and governing law from the contract. Return the information in a JSON format."
    # Basic instruction format - you might refine this
    formatted = f"<start_of_turn>user\n{question_text}<end_of_turn>\n<start_of_turn>model\n{prompt_text}<end_of_turn>\n<start_of_turn>user\n{completion_text}<end_of_turn>"
    return {"text": formatted}

# --- Main Script ---


print("--- Configuration ---")
print(f"Base Model: {MODEL_CONFIG['base_model_name']}")
print(
    f"Max Sequence Length: {MODEL_CONFIG['max_seq_length']} (Truncation Active!)")
print(f"Quantization Bits: {MODEL_CONFIG['quantization_bits']}")
print(f"Train File: {DATA_CONFIG['train_file']}")
print(f"Test File: {DATA_CONFIG['test_file']}")
print(f"Output Directory: {TRAINING_CONFIG['output_dir']}")
print("-" * 20)

# 1. Load Datasets
print("--- Loading Datasets ---")
dataset = load_dataset('json', data_files={
    'train': DATA_CONFIG['train_file'],
    'test': DATA_CONFIG['test_file']
} if os.path.exists(DATA_CONFIG['test_file']) else {
    'train': DATA_CONFIG['train_file']
})

# Optional: Split train if no test file provided
if 'test' not in dataset:
    print("No test file found. Splitting train set for evaluation.")
    dataset = dataset['train'].train_test_split(
        test_size=DATA_CONFIG['test_size'], shuffle=True, seed=42)
else:
    print(
        f"Using provided train ({len(dataset['train'])}) and test ({len(dataset['test'])}) sets.")

train_dataset = dataset['train']
eval_dataset = dataset['test'] if 'test' in dataset else None

print(f"Training samples: {len(train_dataset)}")
if eval_dataset:
    print(f"Evaluation samples: {len(eval_dataset)}")
print("-" * 20)

# 2. Configure Quantization (BitsAndBytes)
print("--- Configuring Quantization ---")
compute_dtype = getattr(
    torch, "bfloat16" if TRAINING_CONFIG['bf16'] else "float16")

bnb_config = None
if MODEL_CONFIG['quantization_bits'] == 4:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # or "fp4"
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,  # Slightly better quality
    )
    print("Using 4-bit QLoRA (nf4 type).")
elif MODEL_CONFIG['quantization_bits'] == 8:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    print("Using 8-bit QLoRA.")
else:
    print("No quantization configured (using full precision).")
print("-" * 20)

# 3. Load Base Model and Tokenizer
print("--- Loading Model and Tokenizer ---")
# Check GPU compatibility with bfloat16
if compute_dtype == torch.bfloat16 and torch.cuda.is_bf16_supported():
    print("BF16 is supported.")
else:
    if compute_dtype == torch.bfloat16:
        print("BF16 not supported, falling back to FP16.")
        compute_dtype = torch.float16
        TRAINING_CONFIG['bf16'] = False
        TRAINING_CONFIG['fp16'] = True


model = AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG['base_model_name'],
    quantization_config=bnb_config,
    device_map="auto",  # Automatically distribute across GPUs if available
    trust_remote_code=True,  # Gemma requires this sometimes
    # attn_implementation="flash_attention_2", # Requires flash-attn library, use if available for speed/memory
    torch_dtype=compute_dtype,  # Load in compute dtype
)
# Disable cache for training stability with gradient checkpointing
model.config.use_cache = False
model.config.pretraining_tp = 1  # Recommendation for PEFT

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CONFIG['base_model_name'], trust_remote_code=True)
# Set padding side for SFT consistency
tokenizer.padding_side = "right"
# Add pad token if missing (Gemma should have it, but good practice)
if tokenizer.pad_token is None:
    print("Tokenizer missing pad token, adding EOS as pad token.")
    tokenizer.pad_token = tokenizer.eos_token

print("Model and Tokenizer loaded.")
# Approximate
print(f"Model Memory Footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
print("-" * 20)


# 4. Configure PEFT (LoRA)
print("--- Configuring LoRA ---")
# Prepare model for k-bit training if using quantization
if bnb_config:
    model = prepare_model_for_kbit_training(model)
    print("Model prepared for k-bit training.")

peft_config = LoraConfig(
    r=QLORA_CONFIG['lora_r'],
    lora_alpha=QLORA_CONFIG['lora_alpha'],
    lora_dropout=QLORA_CONFIG['lora_dropout'],
    target_modules=QLORA_CONFIG['target_modules'],
    bias=QLORA_CONFIG['bias'],
    task_type=QLORA_CONFIG['task_type'],
)

# Apply LoRA adapter to the model
# model = get_peft_model(model, peft_config) # Apply LoRA here *or* let SFTTrainer handle it

# model.print_trainable_parameters() # Verify LoRA layers are trainable
print("LoRA configured.")
print("-" * 20)

# 5. Configure Training Arguments
print("--- Configuring Training Arguments ---")
training_arguments = TrainingArguments(
    output_dir=TRAINING_CONFIG['output_dir'],
    num_train_epochs=TRAINING_CONFIG['num_train_epochs'],
    per_device_train_batch_size=TRAINING_CONFIG['per_device_train_batch_size'],
    per_device_eval_batch_size=TRAINING_CONFIG['per_device_eval_batch_size'],
    gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation_steps'],
    optim=TRAINING_CONFIG['optim'],
    save_steps=TRAINING_CONFIG['save_steps'],
    logging_steps=TRAINING_CONFIG['logging_steps'],
    learning_rate=TRAINING_CONFIG['learning_rate'],
    weight_decay=TRAINING_CONFIG['weight_decay'],
    fp16=TRAINING_CONFIG['fp16'],
    bf16=TRAINING_CONFIG['bf16'],
    max_grad_norm=TRAINING_CONFIG['max_grad_norm'],
    # max_steps=-1, # Set this instead of epochs if you want step-based training
    warmup_ratio=TRAINING_CONFIG['warmup_ratio'],
    group_by_length=TRAINING_CONFIG['group_by_length'],
    lr_scheduler_type=TRAINING_CONFIG['lr_scheduler_type'],
    evaluation_strategy=TRAINING_CONFIG['evaluation_strategy'],
    eval_steps=TRAINING_CONFIG['eval_steps'],
    load_best_model_at_end=TRAINING_CONFIG['load_best_model_at_end'],
    save_total_limit=TRAINING_CONFIG['save_total_limit'],
    report_to=TRAINING_CONFIG['report_to'],
    # Added for potential memory saving with gradient checkpointing
    gradient_checkpointing=True,  # Reduces memory but slows training slightly
    # gradient_checkpointing_kwargs={"use_reentrant": False}, # Recommended for newer torch versions
    # Required by Gemma models when using gradient checkpointing
    # see https://github.com/huggingface/transformers/issues/28339
    use_cache=False,
)
print("Training arguments configured.")
print("-" * 20)


# 6. Initialize SFT Trainer
print("--- Initializing SFT Trainer ---")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,  # Pass LoRA config here - SFTTrainer applies it
    formatting_func=format_prompt,  # Use our formatting function
    # Crucial for managing context length
    max_seq_length=MODEL_CONFIG['max_seq_length'],
    tokenizer=tokenizer,
    args=training_arguments,
    # dataset_text_field="text", # Use this if format_prompt returns a dict with "text" key
    packing=False,  # Set packing=True if you want to combine short sequences - can speed up if many short docs
)
print("SFT Trainer initialized.")
print("Trainable parameters after PEFT applied:")
trainer.model.print_trainable_parameters()  # Verify LoRA layers are trainable
print("-" * 20)


# 7. Start Training
print("--- Starting Training ---")
print(
    f"Effective Batch Size: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
print(f"Using {compute_dtype} compute type.")
if TRAINING_CONFIG['bf16'] or TRAINING_CONFIG['fp16']:
    print("Mixed Precision Training Enabled.")

try:
    train_result = trainer.train()

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("Training finished successfully.")

except Exception as e:
    print(f"An error occurred during training: {e}")
    # Consider adding cleanup or saving state here if needed
    raise e

finally:
    # Ensure model is saved even if there's an interruption or just end of training
    print("--- Saving Final Model ---")
    # This saves the adapter weights, not the full model
    final_adapter_path = os.path.join(
        TRAINING_CONFIG['output_dir'], "final_adapter")
    trainer.save_model(final_adapter_path)
    print(f"Final LoRA adapter saved to {final_adapter_path}")

    # Save tokenizer
    tokenizer.save_pretrained(final_adapter_path)
    print(f"Tokenizer saved to {final_adapter_path}")

    # Optional: Clean up checkpoints if desired, keeping the best/final
    # trainer.cleanup_checkpoints()

    # Optional: Merge adapter into the base model and save full model (requires more disk space & RAM)
    # print("--- Merging Adapter and Saving Full Model ---")
    # try:
    #     # Reload base model in higher precision for merging
    #     base_model_reload = AutoModelForCausalLM.from_pretrained(
    #         MODEL_CONFIG['base_model_name'],
    #         torch_dtype=torch.float16, # Use float16 or float32 for merging
    #         device_map="auto",
    #         trust_remote_code=True,
    #     )
    #     # Load the PEFT adapter
    #     merged_model = PeftModel.from_pretrained(base_model_reload, final_adapter_path)
    #     # Merge the adapter into the base model
    #     merged_model = merged_model.merge_and_unload()

    #     # Save the merged model
    #     merged_model_path = os.path.join(TRAINING_CONFIG['output_dir'], "final_merged_model")
    #     merged_model.save_pretrained(merged_model_path, safe_serialization=True)
    #     tokenizer.save_pretrained(merged_model_path)
    #     print(f"Full merged model saved to {merged_model_path}")
    #     del base_model_reload # Free memory
    #     del merged_model
    #     torch.cuda.empty_cache()
    # except Exception as merge_error:
    #     print(f"Error merging model: {merge_error}. Only adapter was saved.")


print("-" * 20)
print("--- Script Finished ---")


# --- Example Inference (Optional) ---
# print("\n--- Example Inference ---")
# Load the saved adapter for inference
# adapter_path = os.path.join(TRAINING_CONFIG['output_dir'], "final_adapter") # Or path to best checkpoint adapter

# print(f"Loading base model ({MODEL_CONFIG['base_model_name']}) for inference...")
# base_model_inf = AutoModelForCausalLM.from_pretrained(
#     MODEL_CONFIG['base_model_name'],
#     torch_dtype=compute_dtype, # Or torch.float16
#     device_map="auto",
#     trust_remote_code=True,
#     # attn_implementation="flash_attention_2", # Optional speedup
# )
# print(f"Loading adapter from {adapter_path}...")
# model_inf = PeftModel.from_pretrained(base_model_inf, adapter_path)
# model_inf = model_inf.merge_and_unload() # Merge for faster inference (optional)
# model_inf.eval() # Set to evaluation mode

# tokenizer_inf = AutoTokenizer.from_pretrained(adapter_path)
# if tokenizer_inf.pad_token is None:
#     tokenizer_inf.pad_token = tokenizer_inf.eos_token

# # Get a sample prompt from the test set (or define one)
# if eval_dataset and len(eval_dataset) > 0:
#     sample_data = eval_dataset[0]
#     test_prompt_text = sample_data[DATA_CONFIG['prompt_field']]
#     actual_completion = sample_data[DATA_CONFIG['completion_field']]
# else:
#     # Provide a default prompt if no test set available
#     test_prompt_text = "Exhibit 1.3\nAGENCY AGREEMENT\n[...Your sample contract text here...]"
#     actual_completion = "Not available"

# formatted_inference_prompt = f"<start_of_turn>user\n{test_prompt_text}<end_of_turn>\n<start_of_turn>model\n"

# print("\nSample Prompt:")
# print(test_prompt_text[:1000] + "...") # Print first 1000 chars
# print("\nExpected Completion (start):")
# print(str(actual_completion)[:500] + "...") # Print first 500 chars

# print("\nGenerating Completion...")
# pipe = pipeline(task="text-generation", model=model_inf, tokenizer=tokenizer_inf, max_new_tokens=500) # Adjust max_new_tokens
# result = pipe(formatted_inference_prompt)
# generated_text = result[0]['generated_text']

# # Extract only the model's response
# model_response = generated_text.split("<start_of_turn>model\n")[-1]
# # Remove potential end token if present
# if model_response.endswith("<end_of_turn>"):
#     model_response = model_response[:-len("<end_of_turn>")]


# print("\n--- Generated Completion ---")
# print(model_response)
# print("-" * 20)

# print("\n--- Verifying JSON Structure (Optional) ---")
# try:
#     parsed_json = json.loads(model_response)
#     print("Generated text successfully parsed as JSON.")
#     # Add more checks here if needed (e.g., check for specific keys)
# except json.JSONDecodeError:
#     print("WARNING: Generated text is NOT valid JSON.")
# print("-" * 20)
