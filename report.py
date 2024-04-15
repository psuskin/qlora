import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_model(model_name_or_path='huggyllama/llama-7b', adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Fixing some of the early LLaMA HF conversion issues.
    tokenizer.bos_token_id = 1

    # Load the model (use bf16 for faster inference)
    base_model = AutoModelForCausalLM.from_pretrained(
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

    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
    else:
        model = None

    return model, tokenizer

def generate(model, tokenizer, template, prompt, max_new_tokens=2000):
    input = template.format(prompt)
    inputs = tokenizer(input, return_tensors="pt").to('cuda')

    outputs = model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=max_new_tokens)

    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return text.replace(input, '').strip()

if __name__ == "__main__":
    prompts = [
        'Which attributes exist? Context: This is the description of the module "attribut" with the name "Attribute (module)": There are three attribute types in ClassiX®: Preset material characteristic Calculated material characteristic Conditional material characteristic You can find more information in the topic Features. This is the description of the functionality of the module "attribut" with the name "Attribute (module)" regarding Input window: This window is used to maintain the attributes. It varies for the three attribute types, but behaves almost the same. Note: Characteristics are clearly defined via the data field. Therefore, each data field should only be used once, otherwise unwanted results may occur when integrating the attributes within the quotation and order items. This is the description of the functionality of the module "attribut" with the name "Attribute (module)" regarding List window: Serves to list the attribute objects. This is the description of the functionality of the module "attribut" with the name "Attribute (module)" regarding Selection window: This window is used to select an attribute object.',

    ]

    lc_model, lc_tokenizer = load_model('meta-llama/Llama-2-13b-chat-hf')
    _, kal_model, kal_tokenizer = load_model('meta-llama/Llama-2-13b-hf', 'output/klio-alpaca-2-13b-r64-noeval/checkpoint-1875/adapter_model')
    _, karg_model, karg_tokenizer = load_model('meta-llama/Llama-2-13b-hf', 'output/klio-autoregressive-2-13b-r64-noeval/checkpoint-1875/adapter_model')

    implementations = [
        ('LC', 'meta-llama/Llama-2-13b-chat-hf', None, "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{prompt} [/INST]"),
        ('KAL', 'meta-llama/Llama-2-13b-hf', 'output/klio-alpaca-2-13b-r64-noeval/checkpoint-1875/adapter_model', "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat types of reports can I generate using the Order report module?\n\n### Input:\n{prompt}\n\n### Response:"),
        ('KARG', 'meta-llama/Llama-2-13b-hf', 'output/klio-autoregressive-2-13b-r64-noeval/checkpoint-1875/adapter_model', "### Human: {prompt} ### Assistant: ")
    ]

    if os.path.exists("responses.json"):
        with open("responses.json", encoding="utf-8") as f:
            responses = json.load(f)
    else:
        responses = {}

    for implementation in implementations:
        if implementation[0] not in responses:
            responses[implementation[0]] = []

        model, tokenizer = load_model(implementation[0], implementation[1])
        for prompt in prompts:
            if prompt in responses[implementation[0]]:
                continue

            response = generate(model, tokenizer, implementation[2], prompt)
            responses[implementation[0]].append({"query": prompt, "response": response})

    with open("responses.json") as f:
        json.dump(responses, f)