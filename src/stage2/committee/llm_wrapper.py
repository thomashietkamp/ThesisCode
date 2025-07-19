import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ── Configuration ─────────────────────────────────────────────────────────────
_MODEL_ID = "Qwen/Qwen3-14B"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Tokenizer & Model ─────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    _MODEL_ID,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    _MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Optional: a simple HF pipeline for vanilla chat
hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=_DEVICE,
    max_new_tokens=512,
    do_sample=True,
    top_p=0.9
)


# ── Simple chat (no thinking) ───────────────────────────────────────────────────
def chat_complete(prompt: str, temperature: float = 0.3) -> str:
    """
    Vanilla text-completion. Returns the generated text after the prompt.
    """
    resp = hf_pipe(prompt, temperature=temperature,
                   num_return_sequences=1)[0]["generated_text"]
    return resp[len(prompt):].strip()


# ── Chat with “thinking” split ─────────────────────────────────────────────────
def chat_with_thinking(
    messages: list[dict],
    temperature: float = 0.3,
    max_new_tokens: int = 2048
) -> tuple[str, str]:
    """
    Runs the Qwen apply_chat_template with enable_thinking=True.
    Returns: (thinking_content, final_content).

    messages: list of {"role": "user"|"assistant", "content": "..."}
    """
    # 1) Build the full prompt including <think> tags
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    # 2) Tokenize & move to GPU
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[-1]

    # 3) Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

    # 4) Separate thinking vs final tokens
    gen_ids = outputs[0].tolist()[input_len:]
    think_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_id in gen_ids:
        cut = gen_ids.index(think_id)
        thinking_ids = gen_ids[:cut]
        content_ids = gen_ids[cut+1:]
    else:
        thinking_ids = []
        content_ids = gen_ids

    thinking = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
    content = tokenizer.decode(content_ids,  skip_special_tokens=True).strip()
    return thinking, content
