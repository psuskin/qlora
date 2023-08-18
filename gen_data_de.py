import json
import re

import random

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
_RE_COMBINE_PERIOD = re.compile(r"\.+")

toCut = [
    f"""...
(Die Begriffe ClassiX®, CyberEnterprise®, InstantView® und AppsWarehouse® bitte immer mit Warenzeichenhinweis "®" kennzeichnen)""",
]

tooLong = [
    "qmbase",
    "capcheck",
    "genproce",
    "cxItemCostEstimate",

    "cxTxnNote",
    "logfistx",
    "verifydb",

    "cxWorkTimeRule",
    "metamodl",
    "itemVarianceAnalyze",
    "localeEdit",
    "loggidac",
    "resolbom",
    "statturn",

    "item",
    "statpodc",
    "cxWorkFlowRoute",
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
    with open("de_articles.json") as f:
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
    with open("de_articles.json") as f:
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
                "Was ist der Name dieses {textType}s?",
                "Wie heißt dieses {textType}?",
                "Welches {textType} wird beschrieben?",
                "Sage mir den Namen dieses {textType}s.",
                "Benenne dieses {textType}.",
                "Was ist der Name der beschriebenen {textType}s?",
            ],
            "responses": [
                "Der Name dieses {textType}s ist {name}{inMod}.",
                "Dieses {textType} heißt {name}{inMod}.",
                "Das {textType}, welches beschrieben wird, ist {name}{inMod}.",
                "Dieses {textType} nennt sich {name}{inMod}.",
                "Dieses {textType} ist {name}{inMod}.",
            ]
        },

        "describe": {
            "questions": [
                "Was ist der Zweck des {textType}s {name}{inMod}?",
                "Wofür wird das {textType} {name}{inMod} verwendet?",
                "Welchen Zweck erfüllt das {textType} {name}{inMod}?",
                "Beschreibe das {textType} {name}{inMod} für mich.",
                "Erkläre den Zweck des {textType}s {name}{inMod}.",
            ],
            "responses": [
                "Der Zweck des {textType}s {name}{inMod} ist der Folgende: {description}.",
                "Das {textType} {name}{inMod} wird für das Folgende verwendet: {description}.",
                "Das {textType} {name}{inMod} erfüllt den folgenden Zweck: {description}.",
                "Das {textType} {name}{inMod} kann wie folgt beschrieben werden: {description}.",
                "Der Zweck des {textType}s {name}{inMod} ist wie folgt: {description}.",
            ]
        }
    }
    
    with open("de_articles.json") as f:
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

        inputSequence = "\n\n".join([
            "Unten steht eine Anweisung, die eine Aufgabe beschreibt, gepaart mit einer Eingabe, die weiteren Kontext liefert. Schreiben Sie eine Antwort, die die Aufgabe angemessen erfüllt.",
            f"### Anleitung:\n{cleanSequence(random.choice(dataStrings['query']['questions']).format(textType='Modul') + '.')}",
            f"### Kontext:\n{cleanSequence(description + '.')}",
            "### Antwort:"
        ])
        jsonArray.append({"input": inputSequence, "output": cleanSequence(random.choice(dataStrings["query"]["responses"]).format(textType="Modul", name=name, inMod="") + ".")})

        inputSequence = "\n\n".join([
            "Unten steht eine Anweisung, die eine Aufgabe beschreibt. Schreiben Sie eine Antwort, die die Aufgabe angemessen erfüllt.",
            f"### Anleitung:\n{cleanSequence(random.choice(dataStrings['describe']['questions']).format(textType='Modul', name=name, inMod='') + '.')}",
            "### Antwort:"
        ])
        jsonArray.append({"input": inputSequence, "output": cleanSequence(random.choice(dataStrings["describe"]["responses"]).format(textType="Modul", name=name, inMod="", description=description) + ".")})

        for block in article["blocks"]:
            description = cut(block["description"]["text"]).strip()
            if not "Win" in block["name"] or not description:
                continue

            inputSequence = "\n\n".join([
                "Unten steht eine Anweisung, die eine Aufgabe beschreibt, gepaart mit einer Eingabe, die weiteren Kontext liefert. Schreiben Sie eine Antwort, die die Aufgabe angemessen erfüllt.",
                f"### Anleitung:\n{cleanSequence(random.choice(dataStrings['query']['questions']).format(textType='Fenster') + '.')}",
                f"### Kontext:\n{cleanSequence(description + '.')}", 
                "### Antwort:"
            ])
            jsonArray.append({"input": inputSequence, "output": cleanSequence(random.choice(dataStrings["query"]["responses"]).format(textType="Fenster", name=block["name"], inMod=f" in {name}") + ".")})

            inputSequence = "\n\n".join([
                "Unten steht eine Anweisung, die eine Aufgabe beschreibt. Schreiben Sie eine Antwort, die die Aufgabe angemessen erfüllt.",
                f"### Anleitung:\n{cleanSequence(random.choice(dataStrings['describe']['questions']).format(textType='Fenster', name=block['name'], inMod=f' in {name}') + '.')}",
                "### Antwort:"
            ])
            jsonArray.append({"input": inputSequence, "output": cleanSequence(random.choice(dataStrings["describe"]["responses"]).format(textType="Fenster", name=block["name"], inMod=f" in {name}", description=description) + ".")})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonArray, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    #autoregressive("data/de_articles_autoregressive.json")
    alpaca("data/de_articles_alpaca.json")
    #corpus("data/de_articles_corpus.txt")
