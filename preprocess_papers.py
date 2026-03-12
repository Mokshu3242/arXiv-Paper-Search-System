# preprocess_papers.py
import json
import csv
import os
import string
import nltk

nltk.download("stopwords",       quiet=True)
nltk.download("punkt",           quiet=True)
nltk.download("punkt_tab",       quiet=True)
nltk.download("wordnet",         quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem     import WordNetLemmatizer

INPUT_JSON   = "dataset/papers.json"
OUTPUT_JSON  = "dataset/papers_processed.json"
OUTPUT_CSV   = "dataset/papers_processed.csv"

STOP_WORDS  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

def lower_case(text):
    return text.lower()

def remove_punctuation(text):
    clean = ""
    for char in text:
        if char not in string.punctuation:
            clean += char
        else:
            clean += " "
    return clean

def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    filtered = []
    for word in tokens:
        if word not in STOP_WORDS:
            filtered.append(word)
    return filtered

def lemmatize(tokens):
    lemmas = []
    for word in tokens:
        base_word = lemmatizer.lemmatize(word)
        lemmas.append(base_word)
    return lemmas

def remove_short_words(tokens):
    return [word for word in tokens if len(word) > 1]

def clean_text(text):
    print(f"Original:\n {text[:80]}...")

    text   = lower_case(text)
    text   = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    tokens = remove_short_words(tokens)

    print(f"Processed:\n {tokens[:10]}...")
    return tokens

def load_papers(path):
    with open(path, "r", encoding="utf-8") as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers from {path}\n")
    return papers

def process_all_papers(papers):
    processed = []
    for i, paper in enumerate(papers):
        print(f"[{i+1}/{len(papers)}]  {paper['title'][:60]}")
        clean_title = clean_text(paper["title"])
        clean_abstract = clean_text(paper["abstract"])
        combined_tokens = clean_title + clean_abstract
        new_paper = {
            "id":               paper["id"],
            "title":            paper["title"], 
            "abstract":         paper["abstract"],
            "authors":          paper["authors"],
            "year":             paper["year"],
            "category":         paper["category"],
            "url":              paper["url"],
            "clean_title":      clean_title, 
            "clean_abstract":   clean_abstract,
            "all_tokens":       combined_tokens, 
        }
        processed.append(new_paper)
        print()

    return processed

def save_json(papers, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {path}")

def save_csv(papers, path):
    columns = ["id", "title", "abstract", "authors", "year",
               "category", "url", "clean_title", "clean_abstract", "all_tokens"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for paper in papers:
            row = dict(paper)
            row["authors"]        = "; ".join(row["authors"])
            row["clean_title"]    = " ".join(row["clean_title"])
            row["clean_abstract"] = " ".join(row["clean_abstract"])
            row["all_tokens"]     = " ".join(row["all_tokens"])
            writer.writerow(row)
    print(f"Saved CSV: {path}")

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"'{INPUT_JSON}' not found.")
        print("Run collect_papers.py first to build the dataset.")
        return

    print("\nProcessing full dataset …\n")
    papers    = load_papers(INPUT_JSON)
    processed = process_all_papers(papers)

    save_json(processed, OUTPUT_JSON)
    save_csv (processed, OUTPUT_CSV)

    print("\nDone!  Processed files saved to:")
    print(f"     {OUTPUT_JSON}")
    print(f"     {OUTPUT_CSV}")


if __name__ == "__main__":
    main()