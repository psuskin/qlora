import json

def initialImplementation():
    queries = []
    contexts = []
    responses = []

    with open("instructOutput2.txt", encoding="utf-8") as f:
        text = f.readlines()

    currentResponse = ""
    for line in text:
        if line.startswith("Query: "):
            queries.append(line.replace("Query: ", ""))
            if currentResponse:
                responses.append(currentResponse[:-2])
                currentResponse = ""
        elif line.startswith("Context: "):
            contexts.append(line.replace("Context: ", ""))
        elif line.startswith("Response: "):
            currentResponse += line.replace("Response: ", "")
        else:
            currentResponse += f"{line}"
    responses.append(currentResponse[:-2])

    print(len(queries), len(contexts), len(responses))

    samples = []
    samplesNoContext = []
    for query, context, response in zip(queries, contexts, responses):
        if "SAP" in response and not "SAP" in query:
            response.replace("SAP", "classix")

        samples.append({"input": f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{query}\n### Input:\n{context}\n### Response:",
                        "output": response})
        samplesNoContext.append({"input": f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{query}\n### Response:",
                                 "output": response})

    with open("data/en_articles_klio_alpaca.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)

    with open("data/en_articles_klio_alpaca_nocontext.json", "w", encoding="utf-8") as f:
        json.dump(samplesNoContext, f, ensure_ascii=False, indent=4)

def klioImplementation():
    samples = []
    with open("answers.jsonl", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)

            query = sample["question"].strip()
            context = sample["text"].strip()
            response = sample["answer"].strip()
            samples.append({"input": f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{query}\n### Input:\n{context}\n### Response:",
                            "output": response})

    with open("samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    klioImplementation()