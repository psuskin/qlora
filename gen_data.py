import json
import re

import random

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
_RE_COMBINE_PERIOD = re.compile(r"\.+")

toCut = [
    '...   (Please always mark the terms ClassiX®, CyberEnterprise®, InstantView® and AppsWarehouse® with the trademark reference"®")',
    '...   (Please always mark the terms ClassiX®, CyberEnterprise®, InstantView® and AppsWarehouse® with trademark reference"®").',
    '... (Please always mark the terms ClassiX®, CyberEnterprise®, InstantView® and AppsWarehouse® with the trademark reference"®")',
    '...  (Please always mark the terms ClassiX®, CyberEnterprise®, InstantView® and AppsWarehouse® with trademark reference"®").',
    '...  (Please always mark the terms ClassiX®, CyberEnterprise®, InstantView® and AppsWarehouse® with the trademark reference"®")',

    '.....',
    '...',
]

tooLong = [
    "qmbase",
    "capcheck",
    "genproce",
    "cxItemCostEstimate",

    "cxTxnNote",
    "logfistx",
    "verifydb",
]

def cut(string):
    for cutString in toCut:
        string = string.replace(cutString, "")
    
    return string

def cleanSequence(sequence):
    return _RE_COMBINE_PERIOD.sub(".", _RE_COMBINE_WHITESPACE.sub(" ", sequence)).replace("?.", "?").strip()

def joinBlock(name, description):
    if not name or name.isspace() or not description or description.isspace():
        return ""
    else:
        return f"{name.strip()}: {description.strip()}".strip() + ". "

def autoregressive(path):
    with open("en_articles.json") as f:
        articles = json.load(f)

    jsonArray = []

    for article in articles["articles"]:
        sequence = article["module"]
        if article["description"]["text"] and not article["description"]["text"].isspace():
            sequence += ": " + article["description"]["text"].strip()
        sequence += ". "
        
        for block in article["blocks"]:
            sequence += joinBlock(block["name"], block["description"]["text"])

        sequence = cleanSequence(sequence)

        jsonArray.append({"input": "", "output": sequence})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonArray, f, ensure_ascii=False, indent=4)

def corpus(path):
    with open("en_articles.json") as f:
        articles = json.load(f)

    with open(path, "w", encoding="utf-8") as f:
        for article in articles["articles"]:
            sequence = article["module"]
            if article["description"]["text"] and not article["description"]["text"].isspace():
                sequence += ": " + article["description"]["text"].strip()
            sequence += ". "
            
            for block in article["blocks"]:
                sequence += joinBlock(block["name"], block["description"]["text"])

            sequence = cleanSequence(sequence)

            f.write(sequence + "\n")

def alpaca(path, max_words=2000/3):
    dataStrings = {
        "query": {
            "questions": [
                "What is the name of this {textType}?",
                "How is this {textType} called?",
                "Which {textType} is being described?",
                "Tell me the name of this {textType}.",
                "Name this {textType}.",
                "What is the name of the {textType} being described?",
            ],
            "responses": [
                "The name of this {textType} is {name}{inMod}.",
                "This {textType} is called {name}{inMod}.",
                "The {textType} being described is {name}{inMod}.",
                "This {textType} is named {name}{inMod}.",
                "This {textType} is {name}{inMod}.",
            ]
        },

        "describe": {
            "questions": [
                "What is the purpose of the {textType} {name}{inMod}?",
                "What is the {textType} {name}{inMod} used for?",
                "What purpose does the {textType} {name}{inMod} serve?",
                "Describe the {textType} {name}{inMod} for me.",
                "Explain the purpose of the {textType} {name}{inMod}.",
            ],
            "responses": [
                "The purpose of the {textType} {name}{inMod} is as follows: {description}.",
                "The {textType} {name}{inMod} is used for the following: {description}.",
                "The {textType} {name}{inMod} serves the following purpose: {description}.",
                "The {textType} {name}{inMod} can be described as follows: {description}.",
                "The purpose of the {textType} {name}{inMod} is the following: {description}.",
            ]
        }
    }
    
    with open("en_articles.json") as f:
        articles = json.load(f)

    jsonArray = []

    for article in articles["articles"]:
        name = article["module"]
        description = cut(article["description"]["text"]).strip()

        if not description or not article["blocks"]:
            continue

        if name in tooLong:
            continue

        if len(description) < 3:
            print(name, description)
        if len(cleanSequence(description + '.')) < 2:
            continue

        inputSequence = "\n\n".join([
            "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
            f"### Instruction:\n{cleanSequence(random.choice(dataStrings['query']['questions']).format(textType='module') + '.')}",
            f"### Context:\n{cleanSequence(description + '.')}",
            "### Response:"
        ])
        jsonArray.append({"input": inputSequence, "output": cleanSequence(random.choice(dataStrings["query"]["responses"]).format(textType="module", name=name, inMod="") + ".")})

        inputSequence = "\n\n".join([
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
            f"### Instruction:\n{cleanSequence(random.choice(dataStrings['describe']['questions']).format(textType='module', name=name, inMod='') + '.')}",
            "### Response:"
        ])
        jsonArray.append({"input": inputSequence, "output": cleanSequence(random.choice(dataStrings["describe"]["responses"]).format(textType="module", name=name, inMod="", description=description) + ".")})

        for block in article["blocks"]:
            description = cut(block["description"]["text"]).strip()
            if not "Win" in block["name"] or not description:
                continue

            inputSequence = "\n\n".join([
                "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
                f"### Instruction:\n{cleanSequence(random.choice(dataStrings['query']['questions']).format(textType='window') + '.')}",
                f"### Context:\n{cleanSequence(description + '.')}", 
                "### Response:"
            ])
            jsonArray.append({"input": inputSequence, "output": cleanSequence(random.choice(dataStrings["query"]["responses"]).format(textType="window", name=block["name"], inMod=f" in {name}") + ".")})

            inputSequence = "\n\n".join([
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
                f"### Instruction:\n{cleanSequence(random.choice(dataStrings['describe']['questions']).format(textType='window', name=block['name'], inMod=f' in {name}') + '.')}",
                "### Response:"
            ])
            jsonArray.append({"input": inputSequence, "output": cleanSequence(random.choice(dataStrings["describe"]["responses"]).format(textType="window", name=block["name"], inMod=f" in {name}", description=description) + ".")})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonArray, f, ensure_ascii=False, indent=4)

def classification(path):
    with open("en_articles.json") as f:
        articles = json.load(f)

    jsonArray = []

    for article in articles["articles"]:
        name = article["module"]
        description = cut(article["description"]["text"]).strip()

        if not description or not article["blocks"]:
            continue

        if name in tooLong:
            continue

        if len(description) < 3:
            print(name, description)

        inputSequence = cleanSequence(description + '.')
        if len(inputSequence) < 2:
            continue
        jsonArray.append({"input": inputSequence, "label": name})

        for block in article["blocks"]:
            description = cut(block["description"]["text"]).strip()
            if not "Win" in block["name"] or not description:
                continue

            inputSequence = cleanSequence(description + '.')
            jsonArray.append({"input": inputSequence, "label": name})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonArray, f, ensure_ascii=False, indent=4)

def classificationInt(path):
    with open("en_articles.json") as f:
        articles = json.load(f)

    jsonArray = []

    i = 0
    for article in articles["articles"]:
        name = article["module"]
        description = cut(article["description"]["text"]).strip()

        if not description or not article["blocks"]:
            continue

        if name in tooLong:
            continue

        if len(description) < 3:
            print(name, description)

        inputSequence = cleanSequence(description + '.')
        if len(inputSequence) < 2:
            continue
        jsonArray.append({"input": inputSequence, "label": i})
        i += 1

        for block in article["blocks"]:
            description = cut(block["description"]["text"]).strip()
            if not "Win" in block["name"] or not description:
                continue

            inputSequence = cleanSequence(description + '.')
            jsonArray.append({"input": inputSequence, "label": i})
            i += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonArray, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    #autoregressive("data/en_articles_autoregressive.json")
    alpaca("data/en_articles_alpaca.json")
    #corpus("data/en_articles_corpus.txt")
    classification("data/en_articles_classification.json")
    classificationInt("data/en_articles_classification_int.json")