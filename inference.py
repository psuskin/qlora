import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer

def load_model(finetuned=False, model_name_or_path='huggyllama/llama-7b', adapter_path='/workspace/output/guanaco-7b/checkpoint-1875/adapter_model'):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Fixing some of the early LLaMA HF conversion issues.
    tokenizer.bos_token_id = 1

    # Load the model (use bf16 for faster inference)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    )

    if finetuned:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer

def generate(model, tokenizer, prompt, finetuned=False, max_new_tokens=512, top_p=0.9, temperature=0.01):
    if finetuned:
        promptAlpaca = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\nResponse:".format(instruction=prompt)
    else:
        promptAlpaca = prompt
    inputs = tokenizer(promptAlpaca, return_tensors="pt").to('cuda')

    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output = f"\nNetwork output: {text.replace(promptAlpaca, '')}\n"
    return output

if __name__ == "__main__":
    model, tokenizer = load_model(True)

    while (prompt := input("Enter prompt: ")) != "exit":
        print(generate(model, tokenizer, prompt))
