# Instruct
import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

# Inference
from peft import PeftModel
def instruct(promptsPerClass=10):
    instructModel = "meta-llama/Meta-Llama-3-8b-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(instructModel)

    # Load the model (use bf16 for faster inference)
    model = AutoModelForCausalLM.from_pretrained(
        instructModel,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    )

    with open("instructOutput.txt", "r", encoding="utf-8") as f:
        text = f.read()

    text = text.replace("in SAP", "in classix")

    #encountered = False
    descriptionPattern = re.compile(r'^This is the description of', re.MULTILINE)
    instructionPattern = re.compile(r'^\d+\.\s(.+?)(?:\n|$)', re.MULTILINE)
    instructionSets = re.split(descriptionPattern, text)
    for instructionSet in instructionSets:
        if instructionSet:
            instructions = re.findall(instructionPattern, instructionSet)
            for instruction in instructions:
                #if not encountered:
                #    if instruction == 'Are there any manual changes that can be made to the disposition type and/or procurement time of individual parts in the "cxItemDemand" module in SAP?':
                #        encountered = True
                #    continue

                description = "This is the description of" + instructionSet.split("\n")[0]
                prompt = f"""In the following, you will be provided with the description of a module, as well as a query referencing this description which may or may not be answerable based solely on the information provided in the module description. Your task is to assess whether or not the query can be answered with the information provided in the module description. If the query can be answered without additional information, provide the answer. Otherwise, state that the information is not currently available.

Module description: {description}

Query: {instruction}"""
                # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
                llamaPrompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

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
                response = output.split("<|end_header_id|>", 1)[1].strip()

                # Free GPU memory
                #del inputs
                #del outputs
                #torch.cuda.empty_cache()

                #print(llamaPrompt, response)
                with open("instructOutput2.txt", "a", encoding="utf-8") as f:
                    f.write("Query: " + instruction + "\n" + "Context: " + description + "\n" + "Response: " + response + "\n\n")

if __name__ == '__main__':
    instruct()