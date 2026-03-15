# bert_explain.py
import json
import os
import string
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_query(text):
    text = text.lower()
    clean = ""
    for c in text:
        if c not in string.punctuation:
            clean = clean + c
        else:
            clean = clean + " "
    tokens = word_tokenize(clean)
    filtered = []
    for w in tokens:
        if w not in stop_words:
            filtered.append(w)
    lemmatized = []
    for w in filtered:
        lemmatized.append(lemmatizer.lemmatize(w))

    result = []
    for w in lemmatized:
        if len(w) > 1:
            result.append(w)
    return set(result)

def score_label(raw_score):
    if raw_score >= 0.55:
        return "[Strong match]"
    elif raw_score >= 0.35:
        return "[Moderate match]"
    else:
        return "[Weak match -- topic may not exist in dataset]"

def load_resources():
    f = open("dataset/papers_processed.json", encoding="utf-8")
    papers = json.load(f)
    f.close()
    abstracts = []
    titles = []
    urls = []
    all_tokens = []
    paper_ids = []
    for p in papers:
        abstracts.append(p["abstract"])
        titles.append(p["title"])
        urls.append(p.get("url", ""))
        all_tokens.append(set(p["all_tokens"]))
        paper_ids.append(p["id"])
    print("loaded", len(papers), "papers.")
    print()
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    abs_file = "dataset/paper_embeddings.npy"
    title_file = "dataset/title_embeddings.npy"
    if os.path.exists(abs_file):
        abs_embeddings = np.load(abs_file)
    else:
        print("creating abstract embeddings, this might take a while...")
        abs_embeddings = bi_encoder.encode(abstracts, show_progress_bar=True, convert_to_numpy=True)
        np.save(abs_file, abs_embeddings)

    if os.path.exists(title_file):
        title_embeddings = np.load(title_file)
    else:
        print("creating title embeddings, this might take a while...")
        title_embeddings = bi_encoder.encode(titles, show_progress_bar=True, convert_to_numpy=True)
        np.save(title_file, title_embeddings)
    print("embeddings ready, shape:", abs_embeddings.shape)
    print()
    return {
    "abstracts": abstracts,
    "titles": titles,
    "urls": urls,
    "all_tokens": all_tokens,
    "paper_ids": paper_ids,
    "bi_encoder": bi_encoder,
    "cross_encoder": cross_encoder,
    "abs_embeddings": abs_embeddings,
    "title_embeddings": title_embeddings,
}


def run_search(query, top_k, abstracts, titles, urls, all_tokens, bi_encoder, cross_encoder, abs_embeddings, title_embeddings):
    query_tokens = preprocess_query(query)
    query_embedding = bi_encoder.encode(query, convert_to_numpy=True)
    print("preprocessed query tokens:", query_tokens)
    print()
    abs_scores = cosine_similarity([query_embedding], abs_embeddings)[0]
    title_scores = cosine_similarity([query_embedding], title_embeddings)[0]
    bert_scores = 0.6 * abs_scores + 0.4 * title_scores
    candidate_pool = max(50, top_k * 5)
    keyword_candidates = []
    for i in range(len(all_tokens)):
        if query_tokens & all_tokens[i]:
            keyword_candidates.append(i)
    semantic_candidates = np.argsort(bert_scores)[::-1][:candidate_pool]
    candidate_indices = list(set(keyword_candidates) | set(semantic_candidates))
    print("candidate pool size (keyword + semantic):", len(candidate_indices))
    print("reranking with cross-encoder...")
    print()
    ce_inputs = []
    for i in candidate_indices:
        ce_inputs.append([query, titles[i] + " " + abstracts[i][:512]])
    ce_scores = cross_encoder.predict(ce_inputs, show_progress_bar=False)
    reranked_order = np.argsort(ce_scores)[::-1][:top_k]
    final_indices = []
    final_scores = []
    for i in reranked_order:
        final_indices.append(candidate_indices[i])
        final_scores.append(ce_scores[i])
    return final_indices, final_scores, query_tokens

if __name__ == "__main__":
    print("arXiv Paper Search System")
    print("Team Members: Mokshad Sankhe (ms5847), Ishant somal(is488) ") 
    print("-" * 35)
    
    res = load_resources()
    abstracts = res["abstracts"]
    titles = res["titles"]
    urls = res["urls"]
    all_tokens = res["all_tokens"]
    paper_ids = res["paper_ids"]
    bi_encoder = res["bi_encoder"]
    cross_encoder = res["cross_encoder"]
    abs_embeddings = res["abs_embeddings"]
    title_embeddings = res["title_embeddings"]

    query = input("enter search query: ").strip()
    top_k = int(input("how many results do you want? ").strip())
    print()

    final_indices, final_scores, query_tokens = run_search(
        query, top_k,
        abstracts, titles, urls, all_tokens,
        bi_encoder, cross_encoder,
        abs_embeddings, title_embeddings
    )

    print("Results for: '" + query + "'")

    for rank in range(len(final_indices)):
        idx = final_indices[rank]
        ce_score = final_scores[rank]
        matched = query_tokens & all_tokens[idx]

        print()
        print("Rank #" + str(rank + 1))
        print("score (cross-encoder):", round(ce_score, 4), " ", score_label(ce_score))
        print("title  :", titles[idx])
        print("url    :", urls[idx])
        print("preview:", abstracts[idx][:200], "...")

        if matched:
            print("keywords matched:", ", ".join(sorted(matched)))
        else:
            print("explanation: no exact keyword overlap, ranked by semantic similarity.")