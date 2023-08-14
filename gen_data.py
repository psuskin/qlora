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
]

tooLong = [
    "qmbase"
]

def cut(string):
    for cutString in toCut:
        string = string.replace(cutString, "")
    
    return string

def cleanSequence(sequence):
    return _RE_COMBINE_PERIOD.sub(".", _RE_COMBINE_WHITESPACE.sub(" ", sequence)).strip()

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
        description = cut(article["description"]["text"]).strip()

        if not description or not article["blocks"]:
            continue

        if article["module"] in tooLong:
            continue

        sequence = "\n\n".join([
            f"### Context: This is the description of the module {article['module']}: {description}.",
            cleanSequence("### Instruction: " + random.choice(dataStrings["query"]["questions"]).format(textType="module") + "."),
            cleanSequence("### Response: " + random.choice(dataStrings["query"]["responses"]).format(textType="module", name=article["module"], inMod="") + ".")
        ])
        jsonArray.append({"input": "", "output": sequence})

        sequence = "\n\n".join([
            "### Context:",
            cleanSequence("### Instruction: " + random.choice(dataStrings["describe"]["questions"]).format(textType="module", name=article["module"], inMod="") + "."),
            cleanSequence("### Response: " + random.choice(dataStrings["describe"]["responses"]).format(textType="module", name=article["module"], inMod="", description=description) + ".")
        ])
        jsonArray.append({"input": "", "output": sequence})

        for block in article["blocks"]:
            description = cut(block["description"]["text"]).strip()
            if not "Win" in block["name"] or not description:
                continue

            sequence = "\n\n".join([
                cleanSequence(f"### Context: This is the description of the window {block['name']} in module {article['module']}: {description}."),
                cleanSequence("### Instruction: " + random.choice(dataStrings["query"]["questions"]).format(textType="window") + "."),
                cleanSequence("### Response: " + random.choice(dataStrings["query"]["responses"]).format(textType="window", name=block["name"], inMod=f" in {article['module']}") + ".")
            ])
            jsonArray.append({"input": "", "output": sequence})

            sequence = "\n\n".join([
                "### Context:",
                cleanSequence("### Instruction: " + random.choice(dataStrings["describe"]["questions"]).format(textType="window", name=block["name"], inMod=f" in {article['module']}") + "."),
                cleanSequence("### Response: " + random.choice(dataStrings["describe"]["responses"]).format(textType="window", name=block["name"], inMod=f" in {article['module']}", description=description) + ".")
            ])
            jsonArray.append({"input": "", "output": sequence})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonArray, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    #autoregressive("data/en_articles_autoregressive.json")
    alpaca("data/en_articles_alpaca.json")
