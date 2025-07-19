import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.utils as nn_utils
from torch.cuda.amp import autocast, GradScaler
import json
import os
import time
import logging
from pathlib import Path

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dpo_training.log')
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Device setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🔧 Device setup: {device}")
if torch.cuda.is_available():
    logger.info(f"🔧 CUDA version: {torch.version.cuda}")
    logger.info(f"🔧 Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(
            f"🔧 GPU {i}: {props.name} - {props.total_memory / 1e9:.1f}GB")
print(f"Using device: {device}")

# ----------------------------
# Dataset class for DPO training data
# ----------------------------


class DPODataset(Dataset):
    """Dataset class for DPO training data"""

    def __init__(self, jsonl_path: str):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ----------------------------
# Data collator function
# ----------------------------


def smart_truncate_trajectory(prompt, coord_think, coord_out, reason_think, reason_out, valid_think, valid_out, tokenizer, max_length):
    """
    Smart truncation that preserves the most important parts of the trajectory.
    Priority: prompt (full) > outputs (full) > thinking (truncated from middle if needed)
    """
    # Build segments with priorities
    segments = [
        ("[PROMPT]", prompt, 1),  # Highest priority - never truncate
        ("[COORDINATOR OUTPUT]", coord_out, 2),  # High priority
        ("[REASONER OUTPUT]", reason_out, 2),
        ("[VALIDATOR OUTPUT]", valid_out, 2),
        # Lower priority - can truncate
        ("[COORDINATOR THINKING]", coord_think, 3),
        ("[REASONER THINKING]", reason_think, 3),
        ("[VALIDATOR THINKING]", valid_think, 3)
    ]

    # First pass: calculate minimum required tokens for high priority segments
    high_priority_text = ""
    thinking_texts = []

    for tag, text, priority in segments:
        if priority <= 2:  # High priority segments
            high_priority_text += tag + text
        else:  # Thinking segments
            thinking_texts.append((tag, text))

    # Check if high priority segments fit
    high_priority_tokens = tokenizer(
        high_priority_text, return_tensors="pt", truncation=False)
    high_priority_length = high_priority_tokens['input_ids'].shape[1]

    remaining_tokens = max_length - high_priority_length

    if remaining_tokens < 0:
        # Even high priority doesn't fit - use hard truncation
        full_text = "[PROMPT]" + prompt + "[COORDINATOR THINKING]" + coord_think + "[COORDINATOR OUTPUT]" + coord_out + \
            "[REASONER THINKING]" + reason_think + "[REASONER OUTPUT]" + reason_out + \
            "[VALIDATOR THINKING]" + valid_think + \
            "[VALIDATOR OUTPUT]" + valid_out
        return full_text

    # Distribute remaining tokens among thinking segments
    tokens_per_thinking = remaining_tokens // len(
        thinking_texts) if thinking_texts else 0

    # Truncate thinking segments proportionally
    truncated_thinking = []
    for tag, text in thinking_texts:
        if tokens_per_thinking > 50:  # Minimum viable thinking length
            # Estimate characters per token (rough: 1 token ≈ 4 chars)
            max_chars = tokens_per_thinking * 4
            if len(text) > max_chars:
                # Truncate from middle, keeping beginning and end
                keep_start = max_chars // 3
                keep_end = max_chars // 3
                truncated = text[:keep_start] + \
                    " ... [truncated] ... " + text[-keep_end:]
                truncated_thinking.append(tag + truncated)
            else:
                truncated_thinking.append(tag + text)
        else:
            # Very limited space - just add tag with minimal text
            truncated_thinking.append(
                tag + text[:50] + "..." if len(text) > 50 else tag + text)

    # Rebuild trajectory
    result = "[PROMPT]" + prompt + "[COORDINATOR THINKING]" + coord_think + "[COORDINATOR OUTPUT]" + coord_out + \
             "[REASONER THINKING]" + reason_think + "[REASONER OUTPUT]" + reason_out + \
             "[VALIDATOR THINKING]" + valid_think + \
        "[VALIDATOR OUTPUT]" + valid_out

    return result


def dpo_collate_fn(batch):
    """
    Collate function for DPO training data.
    Expected input: List of dicts with keys:
    - prompt, coord_think_plus, coord_out_plus, reason_think_plus, reason_out_plus,
      valid_think_plus, valid_out_plus, coord_think_minus, coord_out_minus,
      reason_think_minus, reason_out_minus, valid_think_minus, valid_out_minus
    """
    collated = {}

    # Get all keys from first item
    keys = batch[0].keys()

    # Collate each field
    for key in keys:
        collated[key] = [item[key] for item in batch]

    return collated


# ----------------------------
# Configuration - Support both local and Modal execution
# ----------------------------

# Check if we're running on Modal (has environment variables set)
is_modal = "DPO_MODEL_NAME" in os.environ

if is_modal:
    # Modal configuration from environment variables
    model_name = os.environ.get("DPO_MODEL_NAME", "Qwen/Qwen3-14B")
    num_epochs = int(os.environ.get("DPO_NUM_EPOCHS", "2"))
    global_batch_size = int(os.environ.get("DPO_GLOBAL_BATCH_SIZE", "16"))
    # Much lower for stability
    learning_rate = float(os.environ.get("DPO_LEARNING_RATE", "1e-5"))
    # Lower gradient clipping
    max_grad_norm = float(os.environ.get("DPO_MAX_GRAD_NORM", "0.5"))
    # Much lower alpha for stability
    alpha = float(os.environ.get("DPO_ALPHA", "0.1"))
    save_steps = int(os.environ.get("DPO_SAVE_STEPS", "500"))

    # Dataset and checkpoint paths from Modal
    train_data_path = os.environ.get(
        "DPO_DATASET_PATH", "/data/dpo_training_data.jsonl")
    val_data_path = train_data_path  # Using same for now
    checkpoint_base_dir = os.environ.get(
        "DPO_CHECKPOINT_DIR", "/data/checkpoints")

    logger.info("🚀 Running on Modal Labs")
    logger.info(f"📊 Using distributed GPU setup")
    logger.info(f"📊 Environment variables loaded:")
    for key, value in os.environ.items():
        if key.startswith("DPO_"):
            logger.info(f"   {key}={value}")

    # Modal-specific device mapping for distributed training
    coord_device_map = {"": 0}  # GPU 0
    reason_device_map = {"": 1}  # GPU 1
    valid_device_map = {"": 2}   # GPU 2

    logger.info(
        f"📊 Device mapping: Coordinator->GPU0, Reasoner->GPU1, Validator->GPU2")

else:
    # Local configuration (original settings)
    model_name = "Qwen/Qwen3-14B"  # Use the base model for DPO training
    num_epochs = 2
    global_batch_size = 16
    learning_rate = 1e-5      # Lower learning rate for stability
    max_grad_norm = 0.5       # Lower gradient clipping
    alpha = 0.1               # Much lower alpha for stability
    save_steps = 500

    # Local dataset paths
    train_data_path = 'src/stage2/data/dpo_training_data.jsonl'
    # TODO: Split this into separate train/val
    val_data_path = 'src/stage2/data/dpo_training_data.jsonl'
    checkpoint_base_dir = './checkpoints'

    logger.info("💻 Running locally")

    # Local device mapping (auto)
    coord_device_map = "auto"
    reason_device_map = "auto"
    valid_device_map = "auto"

# Additional stability parameters
warmup_ratio = 0.1           # Warmup for 10% of training
label_smoothing = 0.0        # No label smoothing for DPO
eps = 1e-8                   # Small epsilon for numerical stability

# Context length settings based on diagnostic results
max_sequence_length = 6144   # Further increased based on 4082 avg token count
truncate_strategy = "smart"  # Can be "hard" or "smart"
logger.info(f"📊 Using max sequence length: {max_sequence_length}")
logger.info(f"📊 Using truncation strategy: {truncate_strategy}")
logger.info("⚠️  Note: Diagnostics showed avg 4082 tokens, using 6144 for safety")

logger.info(f"📋 Training Configuration:")
logger.info(f"   Model: {model_name}")
logger.info(f"   Epochs: {num_epochs}")
logger.info(f"   Global batch size: {global_batch_size}")
logger.info(f"   Per-model batch size: {max(1, global_batch_size // 3)}")
logger.info(f"   Learning rate: {learning_rate}")
logger.info(f"   Alpha (DPO): {alpha}")
logger.info(f"   Max grad norm: {max_grad_norm}")
logger.info(f"   Warmup ratio: {warmup_ratio}")
logger.info(f"   Save steps: {save_steps}")

print(f"Model: {model_name}")
print(f"Epochs: {num_epochs}")
print(f"Batch size: {global_batch_size}")
print(f"Learning rate: {learning_rate}")

# ----------------------------
# 1. Load base models with memory optimization
# ----------------------------

logger.info("🔄 Loading models...")
start_time = time.time()

# Load models with memory optimization
logger.info(f"🔄 Loading Coordinator model on {coord_device_map}...")
coord_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision
    device_map=coord_device_map,  # Device mapping based on environment
    trust_remote_code=True
)
logger.info(f"✅ Coordinator model loaded ({time.time() - start_time:.2f}s)")

logger.info(f"🔄 Loading Reasoner model on {reason_device_map}...")
reason_start = time.time()
reason_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=reason_device_map,
    trust_remote_code=True
)
logger.info(f"✅ Reasoner model loaded ({time.time() - reason_start:.2f}s)")

logger.info(f"🔄 Loading Validator model on {valid_device_map}...")
valid_start = time.time()
valid_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=valid_device_map,
    trust_remote_code=True
)
logger.info(f"✅ Validator model loaded ({time.time() - valid_start:.2f}s)")

# Enable gradient checkpointing for memory efficiency
logger.info("🔄 Enabling gradient checkpointing...")
coord_model.gradient_checkpointing_enable()
reason_model.gradient_checkpointing_enable()
valid_model.gradient_checkpointing_enable()
logger.info("✅ Gradient checkpointing enabled for all models")

logger.info("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, use_fast=False, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
logger.info(f"✅ Tokenizer loaded. Vocab size: {len(tokenizer)}")

logger.info(
    f"🎉 All models loaded successfully! Total time: {time.time() - start_time:.2f}s")

# ----------------------------
# 2. Load datasets
# ----------------------------

# Load the actual DPO training data
logger.info("📊 Loading datasets...")
dataset_start = time.time()
try:
    train_dataset = DPODataset(train_data_path)
    val_dataset = DPODataset(val_data_path)

    logger.info(f"✅ Loaded {len(train_dataset)} training samples")
    logger.info(f"✅ Loaded {len(val_dataset)} validation samples")
    logger.info(f"📊 Dataset loading took {time.time() - dataset_start:.2f}s")

    # Sample a few data points for inspection
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        logger.info(f"📊 Sample data keys: {list(sample.keys())}")
        logger.info(f"📊 Prompt length: {len(sample.get('prompt', ''))}")

    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(val_dataset)} validation samples")

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty!")

except FileNotFoundError as e:
    logger.error(f"❌ Could not find dataset file: {e}")
    print(f"Error: Could not find dataset file: {e}")
    print("Please ensure the DPO training data file exists at the specified path.")
    raise
except Exception as e:
    logger.error(f"❌ Error loading dataset: {e}")
    print(f"Error loading dataset: {e}")
    raise

B = max(1, global_batch_size // 3)  # Ensure at least batch size of 1

# Log batch size calculation details
logger.info(f"📊 Batch size calculation:")
logger.info(f"   Global batch size: {global_batch_size}")
logger.info(f"   Per-model batch size (B): {B}")
logger.info(f"   Total samples per training step: {B * 3}")

if B == 1:
    logger.warning(
        "⚠️  Per-model batch size is 1 - consider increasing global batch size for better statistics")

# ----------------------------
# 3. Inject LoRA (optional)
# ----------------------------
logger.info("🔧 Setting up LoRA adapters...")
lora_start = time.time()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,             # LoRA rank
    lora_alpha=16,
    # typical attention projection targets
    target_modules=["q_proj", "v_proj", "k_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
)

logger.info(
    f"🔧 LoRA config: rank={peft_config.r}, alpha={peft_config.lora_alpha}, dropout={peft_config.lora_dropout}")
logger.info(f"🔧 Target modules: {peft_config.target_modules}")

logger.info("🔧 Applying LoRA to Coordinator model...")
coord_model = get_peft_model(coord_model, peft_config)
logger.info("🔧 Applying LoRA to Reasoner model...")
reason_model = get_peft_model(reason_model, peft_config)
logger.info("🔧 Applying LoRA to Validator model...")
valid_model = get_peft_model(valid_model, peft_config)

# Log trainable parameters
coord_trainable = coord_model.num_parameters(only_trainable=True)
coord_total = coord_model.num_parameters()
logger.info(
    f"✅ Coordinator: {coord_trainable:,} trainable / {coord_total:,} total parameters ({100*coord_trainable/coord_total:.2f}%)")

reason_trainable = reason_model.num_parameters(only_trainable=True)
reason_total = reason_model.num_parameters()
logger.info(
    f"✅ Reasoner: {reason_trainable:,} trainable / {reason_total:,} total parameters ({100*reason_trainable/reason_total:.2f}%)")

valid_trainable = valid_model.num_parameters(only_trainable=True)
valid_total = valid_model.num_parameters()
logger.info(
    f"✅ Validator: {valid_trainable:,} trainable / {valid_total:,} total parameters ({100*valid_trainable/valid_total:.2f}%)")

total_trainable = coord_trainable + reason_trainable + valid_trainable
total_params = coord_total + reason_total + valid_total
logger.info(
    f"🎯 Total: {total_trainable:,} trainable / {total_params:,} total parameters ({100*total_trainable/total_params:.2f}%)")
logger.info(f"✅ LoRA setup completed in {time.time() - lora_start:.2f}s")

# ----------------------------
# 4. Create optimizer that updates ONLY LoRA params
# ----------------------------
optimizers = torch.optim.AdamW(
    list(coord_model.parameters()) +
    list(reason_model.parameters()) +
    list(valid_model.parameters()),
    lr=learning_rate,
    betas=(0.9, 0.98),
    weight_decay=1e-2
)

# ----------------------------
# 5. Setup scheduler and scaler for mixed precision
# ----------------------------
num_training_steps = (len(train_dataset) // global_batch_size) * \
    num_epochs if len(train_dataset) > 0 else 1000
num_warmup_steps = int(num_training_steps * warmup_ratio)
scheduler = get_linear_schedule_with_warmup(
    optimizers,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

logger.info(f"📊 Training schedule:")
logger.info(f"   Total training steps: {num_training_steps}")
logger.info(f"   Warmup steps: {num_warmup_steps}")
logger.info(f"   Steps per epoch: {num_training_steps // num_epochs}")

# Mixed precision scaler
scaler = GradScaler()

# ----------------------------
# 6. Prepare DataLoader
# ----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=B,
    shuffle=True,
    collate_fn=dpo_collate_fn,
    pin_memory=True if device.type == "cuda" else False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=B,
    shuffle=False,
    collate_fn=dpo_collate_fn,
    pin_memory=True if device.type == "cuda" else False
)

# ----------------------------
# 7. Training loop with memory and gradient optimizations
# ----------------------------


def get_device_for_model(model):
    """Get the device where the model is located"""
    if is_modal:
        # On Modal, models are on specific GPUs
        if model == "coord":
            return "cuda:0"
        elif model == "reason":
            return "cuda:1"
        elif model == "valid":
            return "cuda:2"
        else:
            return "cuda:3"  # For aggregation
    else:
        # Local: use the main device
        return device


# Setup checkpoint directory
checkpoint_dir = Path(checkpoint_base_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)


def log_gpu_memory():
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logger.info(
                f"🔧 GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


logger.info("🚀 Starting training loop...")
total_start_time = time.time()

for epoch in range(num_epochs):
    logger.info(f"📊 Epoch {epoch}/{num_epochs-1} starting...")
    epoch_start_time = time.time()
    epoch_loss_sum = 0.0
    epoch_batches = 0

    coord_model.train()
    reason_model.train()
    valid_model.train()

    log_gpu_memory()

    for batch_idx, batch in enumerate(train_loader):
        batch_start_time = time.time()
        logger.info(f"🔄 Processing batch {batch_idx}...")

        optimizers.zero_grad()

        # Use mixed precision
        with autocast():
            # 1. Build inputs for "plus" trajectories
            batch_size = len(batch["prompt"])
            logger.info(f"📊 Batch size: {batch_size}")

            # 1a. Coordinator plus/minus
            input_C_plus = tokenizer(
                ["[PROMPT]" + p + "[COORDINATOR THINKING]" + ct + "[COORDINATOR OUTPUT]" + co
                 for p, ct, co in zip(batch["prompt"], batch["coord_think_plus"], batch["coord_out_plus"])],
                return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length).to(get_device_for_model("coord"))

            input_C_minus = tokenizer(
                ["[PROMPT]" + p + "[COORDINATOR THINKING]" + ct + "[COORDINATOR OUTPUT]" + co
                 for p, ct, co in zip(batch["prompt"], batch["coord_think_minus"], batch["coord_out_minus"])],
                return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length).to(get_device_for_model("coord"))

            # 1b. Reasoner plus/minus
            co_out_plus_texts = ["[PROMPT]" + p + "[COORDINATOR OUTPUT]" + co
                                 for p, co in zip(batch["prompt"], batch["coord_out_plus"])]
            co_out_minus_texts = ["[PROMPT]" + p + "[COORDINATOR OUTPUT]" + co
                                  for p, co in zip(batch["prompt"], batch["coord_out_minus"])]

            input_R_plus = tokenizer(
                [co_text + "[REASONER THINKING]" + rt + "[REASONER OUTPUT]" + ro
                 for co_text, rt, ro in zip(co_out_plus_texts, batch["reason_think_plus"], batch["reason_out_plus"])],
                return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length).to(get_device_for_model("reason"))
            input_R_minus = tokenizer(
                [co_text + "[REASONER THINKING]" + rt + "[REASONER OUTPUT]" + ro
                 for co_text, rt, ro in zip(co_out_minus_texts, batch["reason_think_minus"], batch["reason_out_minus"])],
                return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length).to(get_device_for_model("reason"))

            # 1c. Validator plus/minus
            co_rea_plus = [
                "[PROMPT]" + p + "[COORDINATOR OUTPUT]" + co +
                "[REASONER OUTPUT]" + ro
                for p, co, ro in zip(batch["prompt"], batch["coord_out_plus"], batch["reason_out_plus"])
            ]
            co_rea_minus = [
                "[PROMPT]" + p + "[COORDINATOR OUTPUT]" + co +
                "[REASONER OUTPUT]" + ro
                for p, co, ro in zip(batch["prompt"], batch["coord_out_minus"], batch["reason_out_minus"])
            ]

            input_V_plus = tokenizer(
                [base + "[VALIDATOR THINKING]" + vt + "[VALIDATOR OUTPUT]" + vo
                 for base, vt, vo in zip(co_rea_plus, batch["valid_think_plus"], batch["valid_out_plus"])],
                return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length).to(get_device_for_model("valid"))
            input_V_minus = tokenizer(
                [base + "[VALIDATOR THINKING]" + vt + "[VALIDATOR OUTPUT]" + vo
                 for base, vt, vo in zip(co_rea_minus, batch["valid_think_minus"], batch["valid_out_minus"])],
                return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length).to(get_device_for_model("valid"))

            # 2. Compute log‐probs for each segment (plus/minus)
            def compute_logprob(model, input_dict, model_name):
                forward_start = time.time()
                input_ids = input_dict["input_ids"]
                attention_mask = input_dict["attention_mask"]

                logger.info(
                    f"🔄 {model_name} forward pass: input shape {input_ids.shape}")

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask, return_dict=True)
                logits = outputs.logits  # (B, seq_len, vocab_size)

                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                shift_mask = attention_mask[:, 1:].contiguous()

                # Compute log‐probs with numerical stability
                log_probs = torch.nn.functional.log_softmax(
                    shift_logits, dim=-1)

                # Gather token log-probs
                token_logprobs = torch.gather(
                    log_probs, dim=-1,
                    index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                # Mask padding and sum
                token_logprobs = token_logprobs * shift_mask
                seq_logprob = token_logprobs.sum(dim=1)

                # Normalize by sequence length to prevent length bias
                seq_lengths = shift_mask.sum(dim=1)
                seq_logprob = seq_logprob / (seq_lengths + eps)

                forward_time = time.time() - forward_start
                logger.info(
                    f"✅ {model_name} forward pass completed ({forward_time:.2f}s)")
                logger.info(
                    f"📊 {model_name} avg logprob: {seq_logprob.mean():.4f} ± {seq_logprob.std():.4f}")

                # Move to aggregation device (GPU 3 on Modal, or main device locally)
                return seq_logprob.to(get_device_for_model("aggregation"))

            # Clear cache between model calls to save memory
            torch.cuda.empty_cache() if device.type == "cuda" else None

            logger.info("🔄 Computing Coordinator logprobs...")
            logp_C_plus = compute_logprob(
                coord_model,  input_C_plus, "Coordinator+")
            logp_C_minus = compute_logprob(
                coord_model,  input_C_minus, "Coordinator-")

            torch.cuda.empty_cache() if device.type == "cuda" else None

            logger.info("🔄 Computing Reasoner logprobs...")
            logp_R_plus = compute_logprob(
                reason_model, input_R_plus, "Reasoner+")
            logp_R_minus = compute_logprob(
                reason_model, input_R_minus, "Reasoner-")

            torch.cuda.empty_cache() if device.type == "cuda" else None

            logger.info("🔄 Computing Validator logprobs...")
            logp_V_plus = compute_logprob(
                valid_model,  input_V_plus, "Validator+")
            logp_V_minus = compute_logprob(
                valid_model,  input_V_minus, "Validator-")

            # 3. Sum trajectory log‐probs
            logp_plus = logp_C_plus + logp_R_plus + logp_V_plus
            logp_minus = logp_C_minus + logp_R_minus + logp_V_minus

            # Log individual components
            logger.info(
                f"📊 Coordinator logprobs: plus={logp_C_plus.mean():.4f}, minus={logp_C_minus.mean():.4f}")
            logger.info(
                f"📊 Reasoner logprobs: plus={logp_R_plus.mean():.4f}, minus={logp_R_minus.mean():.4f}")
            logger.info(
                f"📊 Validator logprobs: plus={logp_V_plus.mean():.4f}, minus={logp_V_minus.mean():.4f}")
            logger.info(
                f"📊 Total logprobs: plus={logp_plus.mean():.4f}, minus={logp_minus.mean():.4f}")

            # 4. DPO loss with numerical stability and regularization
            delta = logp_plus - logp_minus

            # Clip delta to prevent extreme values
            delta = torch.clamp(delta, min=-50.0, max=50.0)

            # Compute DPO loss with numerical stability
            scaled_delta = alpha * delta
            # Use numerically stable logsigmoid
            loss = -torch.nn.functional.logsigmoid(scaled_delta).mean()

            # Add multiple regularization terms
            # 1. L2 regularization to prevent collapsed solutions
            l2_reg = 0.01 * (logp_plus.pow(2).mean() +
                             logp_minus.pow(2).mean())

            # 2. Length bias mitigation - penalize preference based purely on length differences
            # Get actual sequence lengths for plus/minus trajectories
            # Move all attention masks to aggregation device for computation
            agg_device = get_device_for_model("aggregation")
            plus_lengths = input_C_plus["attention_mask"].to(agg_device).sum(dim=1).float() + \
                input_R_plus["attention_mask"].to(agg_device).sum(dim=1).float() + \
                input_V_plus["attention_mask"].to(
                    agg_device).sum(dim=1).float()
            minus_lengths = input_C_minus["attention_mask"].to(agg_device).sum(dim=1).float() + \
                input_R_minus["attention_mask"].to(agg_device).sum(dim=1).float() + \
                input_V_minus["attention_mask"].to(
                    agg_device).sum(dim=1).float()

            length_diff = plus_lengths - minus_lengths
            # Penalize strong preference that correlates with length difference
            length_bias_penalty = 0.005 * \
                torch.abs(delta * length_diff /
                          (plus_lengths + minus_lengths + eps)).mean()

            total_loss = loss + l2_reg + length_bias_penalty

            # Detailed logging
            if delta.numel() > 1:
                delta_std = delta.std().item()
                logger.info(
                    f"📊 DPO delta: {delta.mean():.4f} ± {delta_std:.4f} (range: {delta.min():.4f} to {delta.max():.4f})")
            else:
                logger.info(f"📊 DPO delta: {delta.mean():.4f} (single sample)")

            logger.info(f"📊 DPO loss: {loss.item():.4f}")
            logger.info(f"📊 L2 regularization: {l2_reg.item():.4f}")
            logger.info(
                f"📊 Length bias penalty: {length_bias_penalty.item():.4f}")
            logger.info(f"📊 Total loss: {total_loss.item():.4f}")
            logger.info(
                f"📊 Avg length diff (plus-minus): {length_diff.mean().item():.1f} tokens")

            # Compute accuracy for this batch
            batch_correct = (delta > 0).sum().item()
            batch_total = delta.size(0)
            batch_accuracy = batch_correct / batch_total
            logger.info(
                f"📊 Batch preference accuracy: {batch_accuracy:.4f} ({batch_correct}/{batch_total})")

            # Check for potential issues
            if abs(loss.item()) < 1e-6:
                logger.warning(
                    "⚠️  DPO loss near zero - preference may be too strong or saturating")
            elif loss.item() > 100:
                logger.warning(
                    f"⚠️  DPO loss high ({loss.item():.4f}) - checking for numerical instability")

            # Use total_loss for backprop
            loss = total_loss

        # 5. Backprop with gradient scaling and clipping
        logger.info("🔄 Starting backward pass...")
        backward_start = time.time()
        scaler.scale(loss).backward()
        logger.info(
            f"✅ Backward pass completed ({time.time() - backward_start:.2f}s)")

        # Gradient clipping
        logger.info("🔄 Clipping gradients...")
        scaler.unscale_(optimizers)
        grad_norm = nn_utils.clip_grad_norm_(
            list(coord_model.parameters()) +
            list(reason_model.parameters()) +
            list(valid_model.parameters()),
            max_grad_norm
        )
        logger.info(
            f"📊 Gradient norm: {grad_norm:.4f} (clipped to {max_grad_norm})")

        logger.info("🔄 Optimizer step...")
        scaler.step(optimizers)
        scaler.update()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"📊 Learning rate: {current_lr:.2e}")

        batch_time = time.time() - batch_start_time
        epoch_loss_sum += loss.item()
        epoch_batches += 1
        avg_loss = epoch_loss_sum / epoch_batches

        logger.info(f"✅ Batch {batch_idx} completed in {batch_time:.2f}s")
        logger.info(f"📊 Running average loss: {avg_loss:.4f}")

        # Print progress
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            log_gpu_memory()

        # Save checkpoint
        if batch_idx % save_steps == 0 and batch_idx > 0:
            logger.info(
                f"💾 Saving checkpoint at epoch {epoch}, batch {batch_idx}")
            checkpoint_start = time.time()
            print(f"Saving checkpoint at epoch {epoch}, batch {batch_idx}")
            checkpoint_path = checkpoint_dir / \
                f"checkpoint-epoch-{epoch}-batch-{batch_idx}"
            checkpoint_path.mkdir(exist_ok=True)
            coord_model.save_pretrained(checkpoint_path / "coordinator")
            reason_model.save_pretrained(checkpoint_path / "reasoner")
            valid_model.save_pretrained(checkpoint_path / "validator")
            tokenizer.save_pretrained(checkpoint_path / "tokenizer")
            logger.info(
                f"✅ Checkpoint saved in {time.time() - checkpoint_start:.2f}s")

        # Clear cache periodically
        if batch_idx % 5 == 0 and device.type == "cuda":
            logger.info("🧹 Clearing CUDA cache...")
            torch.cuda.empty_cache()

    # End of epoch logging
    epoch_time = time.time() - epoch_start_time
    final_avg_loss = epoch_loss_sum / epoch_batches if epoch_batches > 0 else 0.0
    logger.info(f"📊 Epoch {epoch} completed in {epoch_time:.2f}s")
    logger.info(f"📊 Average loss for epoch: {final_avg_loss:.4f}")
    logger.info(f"📊 Total batches processed: {epoch_batches}")

    # 6. Validation with proper implementation
    logger.info("🔍 Starting validation...")
    val_start_time = time.time()

    coord_model.eval()
    reason_model.eval()
    valid_model.eval()
    correct = total = 0

    with torch.no_grad():
        for batch in val_loader:
            with autocast():
                # Recompute logprobs for validation (same as training)
                batch_size = len(batch["prompt"])

                # Build all inputs (same as training loop)
                input_C_plus = tokenizer(
                    ["[PROMPT]" + p + "[COORDINATOR THINKING]" + ct + "[COORDINATOR OUTPUT]" + co
                     for p, ct, co in zip(batch["prompt"], batch["coord_think_plus"], batch["coord_out_plus"])],
                    return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length).to(get_device_for_model("coord"))

                input_C_minus = tokenizer(
                    ["[PROMPT]" + p + "[COORDINATOR THINKING]" + ct + "[COORDINATOR OUTPUT]" + co
                     for p, ct, co in zip(batch["prompt"], batch["coord_think_minus"], batch["coord_out_minus"])],
                    return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length).to(get_device_for_model("coord"))

                # Build reasoner inputs
                co_out_plus_texts = ["[PROMPT]" + p + "[COORDINATOR OUTPUT]" + co
                                     for p, co in zip(batch["prompt"], batch["coord_out_plus"])]
                co_out_minus_texts = ["[PROMPT]" + p + "[COORDINATOR OUTPUT]" + co
                                      for p, co in zip(batch["prompt"], batch["coord_out_minus"])]

                input_R_plus = tokenizer(
                    [co_text + "[REASONER THINKING]" + rt + "[REASONER OUTPUT]" + ro
                     for co_text, rt, ro in zip(co_out_plus_texts, batch["reason_think_plus"], batch["reason_out_plus"])],
                    return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length).to(get_device_for_model("reason"))
                input_R_minus = tokenizer(
                    [co_text + "[REASONER THINKING]" + rt + "[REASONER OUTPUT]" + ro
                     for co_text, rt, ro in zip(co_out_minus_texts, batch["reason_think_minus"], batch["reason_out_minus"])],
                    return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length).to(get_device_for_model("reason"))

                # Build validator inputs
                co_rea_plus = [
                    "[PROMPT]" + p + "[COORDINATOR OUTPUT]" + co +
                    "[REASONER OUTPUT]" + ro
                    for p, co, ro in zip(batch["prompt"], batch["coord_out_plus"], batch["reason_out_plus"])
                ]
                co_rea_minus = [
                    "[PROMPT]" + p + "[COORDINATOR OUTPUT]" + co +
                    "[REASONER OUTPUT]" + ro
                    for p, co, ro in zip(batch["prompt"], batch["coord_out_minus"], batch["reason_out_minus"])
                ]

                input_V_plus = tokenizer(
                    [base + "[VALIDATOR THINKING]" + vt + "[VALIDATOR OUTPUT]" + vo
                     for base, vt, vo in zip(co_rea_plus, batch["valid_think_plus"], batch["valid_out_plus"])],
                    return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length).to(get_device_for_model("valid"))
                input_V_minus = tokenizer(
                    [base + "[VALIDATOR THINKING]" + vt + "[VALIDATOR OUTPUT]" + vo
                     for base, vt, vo in zip(co_rea_minus, batch["valid_think_minus"], batch["valid_out_minus"])],
                    return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_length).to(get_device_for_model("valid"))

                # Compute logprobs
                def compute_logprob_eval(model, input_dict):
                    input_ids = input_dict["input_ids"]
                    attention_mask = input_dict["attention_mask"]
                    outputs = model(
                        input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                    logits = outputs.logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    shift_mask = attention_mask[:, 1:].contiguous()
                    log_probs = torch.nn.functional.log_softmax(
                        shift_logits, dim=-1)
                    token_logprobs = torch.gather(
                        log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                    token_logprobs = token_logprobs * shift_mask
                    seq_logprob = token_logprobs.sum(dim=1)

                    # Normalize by sequence length (same as training)
                    seq_lengths = shift_mask.sum(dim=1)
                    seq_logprob = seq_logprob / (seq_lengths + eps)

                    return seq_logprob.to(get_device_for_model("aggregation"))

                # Compute all logprobs
                logp_C_plus = compute_logprob_eval(coord_model, input_C_plus)
                logp_C_minus = compute_logprob_eval(coord_model, input_C_minus)

                logp_R_plus = compute_logprob_eval(reason_model, input_R_plus)
                logp_R_minus = compute_logprob_eval(
                    reason_model, input_R_minus)

                logp_V_plus = compute_logprob_eval(valid_model, input_V_plus)
                logp_V_minus = compute_logprob_eval(valid_model, input_V_minus)

                # Sum trajectory log‐probs
                logp_plus = logp_C_plus + logp_R_plus + logp_V_plus
                logp_minus = logp_C_minus + logp_R_minus + logp_V_minus

                # Check if plus is preferred over minus
                correct += (logp_plus > logp_minus).sum().item()
                total += batch_size

    val_accuracy = correct / total if total > 0 else 0.0
    val_time = time.time() - val_start_time

    logger.info(f"✅ Validation completed in {val_time:.2f}s")
    logger.info(
        f"📊 Validation accuracy: {val_accuracy:.4f} ({correct}/{total})")
    print(f"Epoch {epoch} — Validation pairwise accuracy: {val_accuracy:.4f}")

# Training completion logging
total_time = time.time() - total_start_time
logger.info(f"🎉 Training completed! Total time: {total_time:.2f}s")

# Save final models
logger.info("💾 Saving final models...")
final_save_start = time.time()
print("Training completed! Saving final models...")
final_checkpoint = checkpoint_dir / "final"
final_checkpoint.mkdir(exist_ok=True)

logger.info("💾 Saving Coordinator model...")
coord_model.save_pretrained(final_checkpoint / "coordinator")
logger.info("💾 Saving Reasoner model...")
reason_model.save_pretrained(final_checkpoint / "reasoner")
logger.info("💾 Saving Validator model...")
valid_model.save_pretrained(final_checkpoint / "validator")
logger.info("💾 Saving tokenizer...")
tokenizer.save_pretrained(final_checkpoint / "tokenizer")

final_save_time = time.time() - final_save_start
logger.info(f"✅ Final models saved in {final_save_time:.2f}s")
logger.info(f"📁 Models saved to: {final_checkpoint}")

print(f"Final models saved to {final_checkpoint}")
print("🎉 DPO training completed successfully!")

# Final summary
logger.info("📊 Training Summary:")
logger.info(f"   Total training time: {total_time:.2f}s")
logger.info(f"   Total epochs: {num_epochs}")
logger.info(f"   Final validation accuracy: {val_accuracy:.4f}")
logger.info(f"   Models saved to: {final_checkpoint}")
logger.info("🎉 DPO training completed successfully!")
