import json

with open("en_articles.json") as f:
    articles = json.load(f)

modules = []
for article in articles["articles"]:
    modules.append(article["module"])

print(modules)