import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
def main():
    repo_id = "Qwen/Qwen3-0.6B"
    model_dir = snapshot_download(repo_id=repo_id)

    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    llm = LLM(model_dir, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself, "*50,
        "list all prime numbers within 100, "*50,
        "i wish fuck you,baby!!!, "*50,
        "qi wish fuck you,baby!!!, "*50,
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
