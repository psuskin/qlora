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

    for file in os.listdir("data/en_articles_klio"):
        description = open(f"data/en_articles_klio/{file}", encoding="utf-8").read()

        texts = [""]
        for block in description.split("This is the description of"):
            if block:
                block = "This is the description of" + block
                if len(texts[-1].split()) + len(block.split()) < 1000:
                    texts[-1] += block
                else:
                    texts.append(block)

        for text in texts:
            if not text:
                continue

            prompt = f"""In the following, you will be provided with the description of a module. Your task is to generate realistic questions referencing this module description from the perspective of an unfamiliar user who would like to know more about a certain functionality specific to the provided module. Please use both the imperative and interrogative forms and try not to repeat verbs for the questions to maximize variety.

If the information you plan to query seems module-specific, please reference the module name in the query. NEVER REFER TO A MODULE IN A QUESTION WITHOUT USING ITS NAME. This means to never use the word "module" in a question if you are not providing the module name. For example, NEVER use the terms "the module" or "this module" in a question.

ONLY GENERATE QUESTIONS WHICH CAN BE ANSWERED SOLELY USING THE MODULE DESCRIPTION. DO NOT ASK QUESTIONS WHICH REQUIRE ADDITIONAL INFORMATION.

Module description: {text}"""

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
            print(output)
            response = output.split("<|end_header_id|>")[-1].strip()

            # Free GPU memory
            #del inputs
            #del outputs
            #torch.cuda.empty_cache()

            #print(llamaPrompt, response)
            with open("instructOutput.txt", "a", encoding="utf-8") as f:
                f.write(text + "\n" + response + "\n\n")


if __name__ == '__main__':
    instruct()