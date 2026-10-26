def ngrams(text, n=8):
    tokens = text.lower().split()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

corpus = [
    "a 62 year old man with hypertension presents with chest pain and diaphoresis",
    "the patient was started on aspirin and nitroglycerin for suspected acs",
    "ecg showed st elevation in the inferior leads and troponin was elevated",
]
corpus_ngrams = set()
for doc in corpus:
    corpus_ngrams |= ngrams(doc)

test_items = [
    "a 62 year old man with hypertension presents with chest pain and diaphoresis",
    "a 45 year old woman with diabetes presents with polyuria and weight loss",
    "ecg showed st elevation in the inferior leads and troponin was elevated",
]
print(f"{'Test item':<4}{'8-grams':>9}{'Overlap':>9}{'Share':>8}")
for i, item in enumerate(test_items, 1):
    grams = ngrams(item)
    overlap = len(grams & corpus_ngrams)
    share = overlap / len(grams) if grams else 0
    print(f"{i:<4}{len(grams):>9}{overlap:>9}{share:>8.0%}")
