from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                                cache_dir= '/root/autodl-tmp/Model/',)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                                cache_dir= '/root/autodl-tmp/Model/',)
    return model, tokenizer

if __name__ == '__main__':
    load_model()