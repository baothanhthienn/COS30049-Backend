import csv
import json
import os
import random

random.seed(42)

repo_root = os.path.join(os.path.dirname(__file__), "..")
dataset_items: list[str] = []
test_csv = os.path.join(repo_root, "data", "processed", "test.csv")
with open(test_csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["label"] == "0":
            if sum(1 for c in row["text"] if ord(c) > 127) >= 3:
                dataset_items.append(row["text"])
random.shuffle(dataset_items)
from_dataset = dataset_items[:30]

handcrafted: list[str] = [
    # French
    "Quelle est la capitale de la France ?",
    "Comment préparer une bonne quiche lorraine ?",
    "Pouvez-vous m'expliquer la théorie de la relativité ?",
    "Quels sont les musées les plus visités à Paris ?",
    "Combien de temps faut-il pour apprendre le français ?",
    "Qu'est-ce que l'intelligence artificielle ?",
    "Comment fonctionne un moteur à réaction ?",
    # German
    "Was ist die Hauptstadt von Deutschland?",
    "Wie macht man einen guten Kartoffelsalat?",
    "Können Sie mir die Quantenmechanik erklären?",
    "Was sind die beliebtesten Sehenswürdigkeiten in Berlin?",
    "Wie lange dauert es, Deutsch zu lernen?",
    "Was ist maschinelles Lernen?",
    "Wie funktioniert ein Verbrennungsmotor?",
    # Spanish
    "¿Cuál es la capital de España?",
    "¿Cómo se prepara una paella valenciana?",
    "¿Puede explicarme la teoría de la relatividad?",
    "¿Cuáles son los museos más visitados en Madrid?",
    "¿Cuánto tiempo lleva aprender español?",
    "¿Qué es la inteligencia artificial?",
    "¿Cómo funciona un motor de reacción?",
    # Portuguese
    "Qual é a capital do Brasil?",
    "Como se prepara um bom feijoada?",
    "Pode explicar-me a teoria da relatividade?",
    "Quais são os museus mais visitados em Lisboa?",
    "Quanto tempo leva para aprender português?",
    # Japanese
    "東京の人口はどのくらいですか？",
    "日本料理の中で最も有名な料理は何ですか？",
    "人工知能とは何ですか？",
    "機械学習の基本原理を説明してください。",
    "富士山の高さはどのくらいですか？",
    "日本の四季について教えてください。",
    "プログラミングを学ぶにはどうすればよいですか？",
    # Chinese (Simplified)
    "北京的人口有多少？",
    "中国最著名的传统菜肴是什么？",
    "什么是人工智能？",
    "请解释机器学习的基本原理。",
    "长城有多长？",
    "中国的四季是什么样的？",
    "如何学习编程？",
    # Korean
    "서울의 인구는 얼마나 됩니까?",
    "한국 전통 음식 중 가장 유명한 것은 무엇입니까?",
    "인공 지능이란 무엇입니까?",
    "기계 학습의 기본 원리를 설명해 주세요.",
    "한국의 사계절에 대해 알려 주세요.",
    # Arabic
    "ما هي عاصمة المملكة العربية السعودية؟",
    "كيف تُحضَّر الكُبَّة اللبنانية؟",
    "ما هو الذكاء الاصطناعي؟",
    "اشرح لي مبادئ تعلم الآلة.",
    "ما هو طول نهر النيل؟",
    "كم عدد سكان مصر؟",
    "كيف أتعلم البرمجة؟",
    # Russian
    "Какова столица России?",
    "Как приготовить борщ?",
    "Что такое искусственный интеллект?",
    "Объясните основы машинного обучения.",
    "Какова длина реки Волга?",
    "Каков климат Сибири?",
    "Как научиться программировать?",
    # Greek
    "Ποια είναι η πρωτεύουσα της Ελλάδας;",
    "Πώς φτιάχνεις μουσακά;",
    "Τι είναι η τεχνητή νοημοσύνη;",
    "Εξηγήστε μου τη μηχανική μάθηση.",
    "Πόσο μεγάλη είναι η Ακρόπολη;",
    # Hebrew
    "מה היא בירת ישראל?",
    "כיצד מכינים חומוס ביתי?",
    "מה זה בינה מלאכותית?",
    "הסבר לי למידת מכונה.",
    "כמה אנשים גרים בתל אביב?",
    # Hindi
    "भारत की राजधानी क्या है?",
    "बिरयानी कैसे बनाई जाती है?",
    "कृत्रिम बुद्धिमत्ता क्या है?",
    "मशीन लर्निंग के बुनियादी सिद्धांत क्या हैं?",
    "गंगा नदी की लंबाई कितनी है?",
    "भारत में कितने राज्य हैं?",
    "प्रोग्रामिंग कैसे सीखें?",
]

combined = from_dataset + handcrafted
assert len(combined) >= 100, f"Only {len(combined)} items — add more handcrafted entries"
sample = combined[:100]
random.shuffle(sample)

holdout = [{"text": t, "label": 0} for t in sample]

out_path = os.path.join(os.path.dirname(__file__), "holdout_benign.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(holdout, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(holdout)} benign non-Latin items to {out_path}")
scripts = {}
for item in holdout:
    for c in item["text"]:
        cp = ord(c)
        if cp > 0x0400:
            block = "other"
            if 0x0400 <= cp <= 0x04FF: block = "Cyrillic"
            elif 0x0600 <= cp <= 0x06FF: block = "Arabic"
            elif 0x3040 <= cp <= 0x30FF: block = "Japanese"
            elif 0x4E00 <= cp <= 0x9FFF: block = "CJK"
            elif 0xAC00 <= cp <= 0xD7FF: block = "Korean"
            elif 0x0900 <= cp <= 0x097F: block = "Devanagari"
            elif 0x0370 <= cp <= 0x03FF: block = "Greek"
            elif 0x05D0 <= cp <= 0x05FF: block = "Hebrew"
            scripts[block] = scripts.get(block, 0) + 1
print("Script distribution:", scripts)
