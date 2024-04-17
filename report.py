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
        model = base_model

    return model, tokenizer

def generate(model, tokenizer, template, prompt, max_new_tokens=2000):
    input = template.format(prompt=prompt)
    inputs = tokenizer(input, return_tensors="pt").to('cuda')

    outputs = model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=max_new_tokens, do_sample=False)

    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return text.replace(input.replace('<s>', ''), '').strip()

if __name__ == "__main__":
    prompts = [
        'Which attributes exist? Context: This is the description of the module "attribut" with the name "Attribute (module)": There are three attribute types in ClassiX®: Preset material characteristic Calculated material characteristic Conditional material characteristic You can find more information in the topic Features. This is the description of the functionality of the module "attribut" with the name "Attribute (module)" regarding Input window: This window is used to maintain the attributes. It varies for the three attribute types, but behaves almost the same. Note: Characteristics are clearly defined via the data field. Therefore, each data field should only be used once, otherwise unwanted results may occur when integrating the attributes within the quotation and order items. This is the description of the functionality of the module "attribut" with the name "Attribute (module)" regarding List window: Serves to list the attribute objects. This is the description of the functionality of the module "attribut" with the name "Attribute (module)" regarding Selection window: This window is used to select an attribute object.',
        "What is a Gozintograph? Context: This is the description of the module \"cxItemDemand\" with the name \"Parts requirement (Gozintograph)\": Starting from parts scheduling, the production-specific parts lists selected there - sorted according to production stages - are summarised in so-called gozintographs. The parts request module(business pattern) described here is used to maintain such - automatically created - gozintographs.The task of this module is to be able to carry out a purely logistical material planning in a first step. This is done in six steps:Determination of the disposition type of the partsDetermination of the standard replenishment lead time according to the disposition typeScheduling of the necessary provision of parts Checking the availability of materials Triggering of production orders, purchase requisitions and stock reservations Release of the parts requirementLink with ex ante needs Quantity changeDelete direct successor documents Steps 1, 2 and 3 can be called up and carried out directly and automatically as one process from parts planning. The \"standard information\" on the procurement type and time from the master information of the parts master is used as the basis. Steps 3 and 4 (planning) may have to be carried out iteratively several times after manually changing the disposition type and/or procurement time of individual parts so that the planned dates can be met. This planning can also be carried out graphically supported by means of a Gantt chart.",


        "What is a variant part?",
        "What is a variant part in classix?"
    ]

    implementations = [
        ('LC', 'meta-llama/Llama-2-13b-chat-hf', None, "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{prompt} [/INST]"),
        ('KAL', 'meta-llama/Llama-2-13b-hf', 'output/klio-alpaca-2-13b-r64-noeval/checkpoint-1875/adapter_model', "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"),
        ('KARG', 'meta-llama/Llama-2-13b-hf', 'output/klio-autoregressive-2-13b-r64-noeval/checkpoint-1875/adapter_model', "### Human: {prompt}### Assistant: ")
    ]

    if os.path.exists("responses.json"):
        with open("responses.json", encoding="utf-8") as f:
            responses = json.load(f)
    else:
        responses = {}

    for implementation in implementations:
        name = implementation[0]
        if name not in responses:
            responses[name] = []

        if set(prompts) == set([response["query"] for response in responses[name]]):
            continue

        model, tokenizer = load_model(implementation[1], implementation[2])
        for prompt in prompts:
            if prompt in [response["query"] for response in responses[name]]:
                continue

            print(f"Computing response for {name}...")
            response = generate(model, tokenizer, implementation[3], prompt)
            responses[name].append({"query": prompt, "response": response})
            #print(prompt)
            #print(implementation[3])
            #print(response)

        del model
        del tokenizer
        torch.cuda.empty_cache()

    with open("responses.json", "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)
