# kv_estimate.py
import torch
from transformers import AutoConfig

model_id = "Qwen/Qwen3-0.6B"

cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

L = int(getattr(cfg, "num_hidden_layers"))
n_heads = int(getattr(cfg, "num_attention_heads"))
n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
hidden = int(getattr(cfg, "hidden_size"))
head_dim = hidden // n_heads

dtype_bytes = 2  # fp16/bf16
bytes_per_token = 2 * L * n_kv * head_dim * dtype_bytes

print("layers =", L)
print("n_heads =", n_heads, "n_kv_heads =", n_kv, "head_dim =", head_dim)
print("KV bytes/token =", bytes_per_token, f"(~{bytes_per_token/1024:.1f} KiB)")

if torch.cuda.is_available():
    free_b, total_b = torch.cuda.mem_get_info()
    print("CUDA free =", free_b/1024**3, "GiB; total =", total_b/1024**3, "GiB")
    est_tokens = int(free_b / bytes_per_token)
    print("Rough max KV tokens (free_mem / bytes_per_token) =", est_tokens)
else:
    print("CUDA not available; can't estimate free mem.")
