import json
import re

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
_RE_COMBINE_PERIOD = re.compile(r"\.+")

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
    return _RE_COMBINE_PERIOD.sub(".", _RE_COMBINE_WHITESPACE.sub(" ", sequence)).strip()

with open("en_articles.json") as f:
    articles = json.load(f)

jsonArray = []

toCut = [
    '(Please always mark the terms ClassiX®, CyberEnterprise®, InstantView® and AppsWarehouse® with the trademark reference\"®\")',
    '(Please always mark the terms ClassiX®, CyberEnterprise®, InstantView® and AppsWarehouse® with trademark reference\"®\")',
    '...',
]
def cutDescription(description):
    for cutStr in toCut:
        description = description.replace(cutStr, "")

    return description.strip()

for article in articles["articles"]:
    if (not article["name"] or article["name"].isspace()) or len(article["blocks"]) < 1:
        continue

    sequence = article["name"]
    description = cutDescription(article["description"]["text"])
    if description and not description.isspace():
        sequence += ": " + description
    if sequence.endswith("."):
        sequence += " "
    else:
        sequence += ". "
    for block in article["blocks"]:
        sequence += joinBlock(block["name"], block["description"]["text"])

    sequence = cleanSequence(sequence)

    jsonArray.append({"input": sequence, "output": ""})

with open("data/en_articles_autoregressive_input.json", "w", encoding="utf-8") as f:
    json.dump(jsonArray, f, ensure_ascii=False, indent=4)
