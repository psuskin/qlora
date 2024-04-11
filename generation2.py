# Instruct
import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

# Inference
from peft import PeftModel

EN = True

def instruct(promptsPerClass=10):
    if EN:
        instructModel = "meta-llama/Llama-2-13b-chat-hf"

        tokenizer = AutoTokenizer.from_pretrained(instructModel)
        # Fixing some of the early LLaMA HF conversion issues.
        tokenizer.bos_token_id = 1

        # Load the model (use bf16 for faster inference)
        model = AutoModelForCausalLM.from_pretrained(
            instructModel,
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
    else:
        instructModel = "jphme/Llama-2-13b-chat-german"

        tokenizer = AutoTokenizer.from_pretrained(instructModel)
        tokenizer.pad_token_id=tokenizer.eos_token_id

        # Load the model (use bf16 for faster inference)
        model = AutoModelForCausalLM.from_pretrained(
            instructModel,
            device_map={"": 0},
            load_in_4bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
        )

    with open("instructOutput.txt", "r", encoding="utf-8") as f:
        text = f.read()

    pattern = re.compile(r'\d+\.\s(.+?)(?:\n|$)')
    instructionSets = text.split("This is the description of")
    for instructionSet in instructionSets:
        if instructionSet:
            instructions = re.findall(pattern, instructionSet)
            for instruction in instructions:
                description = "This is the description of" + instructionSet.split("\n")[0]
                prompt = f"""In the following, you will be provided with the description of a module, as well as a query referencing this description which may or may not be answerable based solely on the information provided in the module description. Your task is to assess whether or not the query can be answered with the information provided in the module description. If the query can be answered without additional information, provide the answer. Otherwise, state that the information is not currently available.

Module description: {description}

Query: {instruction}"""
                llamaPrompt = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{prompt} [/INST]"""

                inputs = tokenizer(llamaPrompt, return_tensors="pt").to('cuda')

                outputs = model.generate(
                    **inputs,
                    generation_config=GenerationConfig(
                        do_sample=True,
                        max_new_tokens=4096,
                        top_p=1,
                        temperature=0.01,
                        repetition_penalty=1.2
                    )
                )

                output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = output.split("[/INST]" if EN else "ASSISTANT:", 1)[1].strip()

                # Free GPU memory
                #del inputs
                #del outputs
                #torch.cuda.empty_cache()

                #print(llamaPrompt, response)
                with open("instructOutput2.txt", "a", encoding="utf-8") as f:
                    f.write("Query: " + instruction + "\n" + "Context: " + description + "\n" + "Response: " + response + "\n\n")

if __name__ == '__main__':
    instruct()

    # inference("distilbert-base-uncased-classification/checkpoint-30000")