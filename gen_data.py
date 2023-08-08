import json
import re

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

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

with open("data/en_articles_autoregressive.json", "w", encoding="utf-8") as f:
    json.dump(jsonArray, f, ensure_ascii=False, indent=4)