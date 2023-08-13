import json
import re

import random

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

toCut = [
    '...   (Please always mark the terms ClassiX®, CyberEnterprise®, InstantView® and AppsWarehouse® with the trademark reference"®")',
    '...   (Please always mark the terms ClassiX®, CyberEnterprise®, InstantView® and AppsWarehouse® with trademark reference"®").',
    '... (Please always mark the terms ClassiX®, CyberEnterprise®, InstantView® and AppsWarehouse® with the trademark reference"®")',
    '...  (Please always mark the terms ClassiX®, CyberEnterprise®, InstantView® and AppsWarehouse® with trademark reference"®").',
    '...  (Please always mark the terms ClassiX®, CyberEnterprise®, InstantView® and AppsWarehouse® with the trademark reference"®")',
]

def cut(string):
    for cutString in toCut:
        string = string.replace(cutString, "")
    
    return string

def joinBlock(name, description):
    if not name or name.isspace() or not description or description.isspace():
        return ""
    else:
        joinedBlock = f"{name.strip()}: {description.strip()}".strip()
        if joinedBlock.endswith("."):
            return joinedBlock + " "
        else:
            return joinedBlock + ". "

def cleanSequence(sequence):
    return _RE_COMBINE_WHITESPACE.sub(" ", sequence).strip()

def autoregressive(path):
    with open("en_articles.json") as f:
        articles = json.load(f)

    jsonArray = []

    for article in articles["articles"]:
        sequence = article["name"]
        if article["description"]["text"] and not article["description"]["text"].isspace():
            sequence += ": " + article["description"]["text"].strip()
        if sequence.endswith("."):
            sequence += " "
        else:
            sequence += ". "
        
        for block in article["blocks"]:
            sequence += joinBlock(block["name"], block["description"]["text"])

        sequence = cleanSequence(sequence)

        jsonArray.append({"input": "", "output": sequence})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonArray, f, ensure_ascii=False, indent=4)

def alpaca(path):
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
                "The {textType} {name}{inMod} serves the purpose of the following: {description}.",
                "The {textType} {name}{inMod} can be described as follows: {description}.",
                "The purpose of the {textType} {name}{inMod} is the following: {description}.",
            ]
        }
    }
    
    with open("en_articles.json") as f:
        articles = json.load(f)

    jsonArray = []

    for article in articles["articles"]:
        description = cut(article["description"]["text"]).strip()
        if "trademark reference" in description:
            print(description)
        if not description:
            continue

        sequence = f"### Context: This is the description of {article['name']}: {description}"
        if sequence.endswith("."):
            sequence += "\n\n"
        else:
            sequence += ".\n\n"
        sequence += "### Instruction: " + random.choice(dataStrings["query"]["questions"]).format(textType="module") + "\n\n"
        sequence += "### Response: " + random.choice(dataStrings["query"]["responses"]).format(textType="module", name=article["name"], inMod="")
        jsonArray.append({"input": "", "output": cleanSequence(sequence)})

        sequence = "### Context:\n\n"
        sequence += "### Instruction: " + random.choice(dataStrings["describe"]["questions"]).format(textType="module", name=article["name"], inMod="") + "\n\n"
        sequence += "### Response: " + random.choice(dataStrings["describe"]["responses"]).format(textType="module", name=article["name"], inMod="", description=description)
        if sequence.endswith("."):
            sequence += "\n\n"
        else:
            sequence += ".\n\n"
        jsonArray.append({"input": "", "output": cleanSequence(sequence)})

        for block in article["blocks"]:
            description = cut(block["description"]["text"]).strip()
            if not "Win" in block["name"] or not description:
                continue

            sequence += f"### Context: This is the description of {block['name']} in {article['name']}: {description}"
            if sequence.endswith("."):
                sequence += "\n\n"
            else:
                sequence += ".\n\n"
            sequence += "### Instruction: " + random.choice(dataStrings["query"]["questions"]).format(textType="window") + "\n\n"
            sequence += "### Response: " + random.choice(dataStrings["query"]["responses"]).format(textType="window", name=block["name"], inMod=f" in {article['name']}")
            jsonArray.append({"input": "", "output": cleanSequence(sequence)})

            sequence = "### Context:\n\n"
            sequence += "### Instruction: " + random.choice(dataStrings["describe"]["questions"]).format(textType="window", name=block["name"], inMod=f" in {article['name']}") + "\n\n"
            sequence += "### Response: " + random.choice(dataStrings["describe"]["responses"]).format(textType="window", name=block["name"], inMod=f" in {article['name']}", description=description)
            if sequence.endswith("."):
                sequence += "\n\n"
            else:
                sequence += ".\n\n"
            jsonArray.append({"input": "", "output": cleanSequence(sequence)})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonArray, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    #autoregressive("data/en_articles_autoregressive.json")
    alpaca("data/en_articles_alpaca.json")