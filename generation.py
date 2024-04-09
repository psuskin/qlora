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

    prompts = []
    for file in os.listdir("data/en_articles_klio"):
        text = open(f"data/en_articles_klio/{file}", encoding="utf-8").read()
        
        if EN:
            prompt = f"""In the following, I will provide you with the description of a module. Please generate {promptsPerClass} numbered question-answer pairs of the format 1. Q: question A: answer, where the question queries some relevant aspect of the module's functionality and the answer provides an accurate and helpful response to the corresponding question.

Here are the requirements:
- The questions should be based on the module description.
- The answers should be accurate and helpful.
- The answers should not contain any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
- The responses should be socially unbiased and positive in nature.
- If a question is not factually coherent, explain why instead of providing an incorrect answer.
- If you do not know the answer to a question, please do not provide false information.
- The questions should be varied and cover different aspects of the module's functionality.
- Either an imperative or interrogative form can be used for the questions.
- Try not to repeat the verb for the questions to maximize variety.

Module description: {text}"""
        else:
            prompt = f"Ich gebe dir im Folgenden eine Beschreibung eines einzelnen Moduls. Bitte generiere {promptsPerClass} Frage-Antwort Paare, bei denen die Frage einen relevanten Aspekt der Funktionalität des Moduls abfragt und die Antwort eine genaue und hilfreiche Reaktion zur entsprechenden Frage bietet.\n\nModulbeschreibung: {text}"
        prompts.append(prompt)
    
    instructions = []
    for prompt in prompts:
        if EN:
            # https://huggingface.co/blog/llama2
            llamaPrompt = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{prompt} [/INST]"""
        else:
            llamaPrompt = f"Du bist ein hilfreicher Assistent. USER: {prompt} ASSISTANT:"

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

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = text.split("[/INST]" if EN else "ASSISTANT:", 1)[1].strip()

        #print(llamaPrompt, response)
        with open("instructOutput.txt", "a", encoding="utf-8") as f:
            f.write(llamaPrompt + "\n" + response + "\n\n")
        # exit()

        #pattern = re.compile(r'\d+\.\s(.+?)(?:\n|$)')
        #matches = pattern.findall(response)
        #for match in matches:
        #    pair = match.split("A:")
        #    if len(pair) == 2:
        #        instructions.append({"input": "In the following is a query. Write an appropriate response.\n\n### Query: " + pair[0].replace("Q:", "").strip().strip("\"") + "\n\n###Response:", "output": pair[1].strip().strip("\"")})

    #with open(f"data/{'en' if EN else 'de'}_articles_generation_instruct{promptsPerClass}.json", "w", encoding="utf-8") as f:
    #    json.dump(instructions, f, ensure_ascii=False, indent=4)

def inference(modelName, adapter_path):
    prompts = [
        ("Which module provides version and copyright information?", 0),

        ("How can I calculate the current time in another location?", 627),
        ("How can I calculate the current time in another location while accounting for discrepancies due to time zones?", 627),
        ("With which module can I calculate the current time in another location while accounting for discrepancies due to time zones?", 627),

        ("Tell me how to test the conversion of a temperature into the different heat units.", 460),
        ("Where do I record both flexitime and operating data (BDE)?", 626),
        ("Where can I check offer/order data?", 608),
        ("Help me with inspection of partner data.", 609),
        ("Provide me with resources on inspection of purchasing data.", 610),

        ("Parts lists describe the composition of a production part. A bill of material consists of parts, which in turn can have a bill of material.", 45),
        ("Which module am I referencing? Parts lists describe the composition of a production part. A bill of material consists of parts, which in turn can have a bill of material.", 45),
        
        ("Welches Modul ist dafür verantwortlich, Informationen über Versionen und Urheberrecht anzuzeigen?", 0),
        ("Anzeige und Auflistung der Versions- und Urheberrechtsinformationen.", 0),
    ]

    tokenizer = AutoTokenizer.from_pretrained(modelName)
    # Fixing some of the early LLaMA HF conversion issues.
    tokenizer.bos_token_id = 1

    # Load the model (use bf16 for faster inference)
    base_model = AutoModelForCausalLM.from_pretrained(
        modelName,
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

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    for prompt, _ in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

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

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = text.replace(prompt, "").strip()

        print(prompt, response)

if __name__ == '__main__':
    instruct()

    # inference("distilbert-base-uncased-classification/checkpoint-30000")