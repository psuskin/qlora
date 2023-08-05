import json

with open("en_articles.json") as f:
    articles = json.load(f)

print(list(articles["articles"][0].keys()))

# with open("data/en_articles_text_module.json", "w", encoding="utf-8") as f:
#     for article in articles["articles"]:
#         description = article['text'].replace('"', '\\"').replace('\\\\"', '\\"')
#         f.write(f"{{\"input\": \"{description}\", \"output\": \"{article['module']}\"}}\n")

with open("data/en_articles_text_module.tsv", "w", encoding="utf-8") as f:
    f.write("input\toutput\n")
    for article in articles["articles"]:
        description = article['text'].replace('"', '\\"').replace('\\\\"', '\\"').replace('\n', ' ')
        if '\t' in description:
            continue
        f.write(f"{description}\t{article['module']}\n")