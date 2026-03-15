# evaluate.py
import json
import os
import string
import math
import csv
import time
import numpy as np
import nltk
from dotenv import load_dotenv

load_dotenv()

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

from bm25_explain import connect, get_term_details, index_name
from bert_explain import load_resources, preprocess_query, score_label
from hybrid import bm25_search, get_bert_candidates, hybrid_search, normalise

k_values = [1, 3, 5, 10]
alpha = 0.4
beta = 0.6
judgements_file = "manual_judgements.json"

test_queries = [
    "transformer attention mechanism",
    "generative adversarial network image synthesis",
    "reinforcement learning reward function",
    "object detection convolutional neural network",
    "natural language processing text classification",
    "graph neural network node classification",
    "federated learning privacy distributed",
    "knowledge graph embedding relation",
    "image segmentation semantic pixel",
    "recurrent neural network sequence modeling",
    "contrastive learning self supervised representation",
    "neural machine translation encoder decoder",
    "anomaly detection unsupervised",
    "diffusion model image generation",
    "large language model fine tuning instruction",
    "transfer learning domain adaptation",
    "bayesian optimization hyperparameter tuning",
    "speech recognition acoustic model",
    "neural architecture search automl",
    "point cloud 3d deep learning",
    "multi task learning shared representation",
    "question answering reading comprehension",
    "pruning quantization model compression",
    "vision transformer image recognition",
    "meta learning few shot",
]


def precision_at_k(relevance, k):
    if k == 0:
        return 0.0
    return sum(relevance[:k]) / k


def ndcg_at_k(relevance, k):
    def dcg(rel, k):
        total = 0.0
        for i in range(min(k, len(rel))):
            total = total + rel[i] / math.log2(i + 2)
        return total
    ideal = dcg(sorted(relevance, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg(relevance, k) / ideal


def run_bm25_search(es, query, top_k):
    raw_hits = bm25_search(es, query, top_k)
    results = []
    for h in raw_hits:
        term_dets = get_term_details(h["explanation"])
        title_score = 0.0
        abstract_score = 0.0
        for t in term_dets:
            if t["field"] == "title":
                title_score = title_score + t["final"]
            if t["field"] == "abstract":
                abstract_score = abstract_score + t["final"]
        results.append({
            "paper_id": h["paper_id"],
            "title": h["source"]["title"],
            "abstract": h["source"]["abstract"],
            "bm25_score": round(h["bm25_raw"], 4),
            "term_details": term_dets,
            "title_score": round(title_score, 4),
            "abstract_score": round(abstract_score, 4),
        })
    return results


def run_semantic_search(resources, query, top_k):
    from sklearn.metrics.pairwise import cosine_similarity
    query_tokens = preprocess_query(query)
    bi = resources["bi_encoder"]
    ce = resources["cross_encoder"]
    q_emb = bi.encode(query, convert_to_numpy=True)
    abs_sim = cosine_similarity([q_emb], resources["abs_embeddings"])[0]
    title_sim = cosine_similarity([q_emb], resources["title_embeddings"])[0]
    combined = 0.6 * abs_sim + 0.4 * title_sim

    kw_cands = []
    for i in range(len(resources["all_tokens"])):
        if query_tokens & resources["all_tokens"][i]:
            kw_cands.append(i)

    sem_cands = list(np.argsort(combined)[::-1][:50])
    cands = list(set(kw_cands) | set(sem_cands))

    ce_inputs = []
    for i in cands:
        ce_inputs.append([query, resources["titles"][i] + " " + resources["abstracts"][i][:512]])
    ce_scores = ce.predict(ce_inputs, show_progress_bar=False)

    order = np.argsort(ce_scores)[::-1][:top_k]
    results = []
    for idx in order:
        i = cands[idx]
        matched = query_tokens & resources["all_tokens"][i]
        results.append({
            "paper_id": resources["paper_ids"][i],
            "title": resources["titles"][i],
            "abstract": resources["abstracts"][i],
            "ce_score": round(float(ce_scores[idx]), 4),
            "abs_sim": round(float(abs_sim[i]), 4),
            "title_sim": round(float(title_sim[i]), 4),
            "matched_tokens": sorted(matched),
        })
    return results


def run_hybrid_search(es, resources, query, top_k):
    return hybrid_search(
        es, query, top_k,
        resources["abstracts"],
        resources["titles"],
        resources["urls"],
        resources["all_tokens"],
        resources["paper_ids"],
        resources["bi_encoder"],
        resources["cross_encoder"],
        resources["abs_embeddings"],
        resources["title_embeddings"],
    )


def load_judgements():
    if os.path.exists(judgements_file):
        f = open(judgements_file, encoding="utf-8")
        content = f.read().strip()
        f.close()
        if not content:
            return {}
        data = json.loads(content)
        return data
    return {}


def save_judgements(j):
    f = open(judgements_file, "w", encoding="utf-8")
    json.dump(j, f, indent=2, ensure_ascii=False)
    f.close()


def get_manual_relevance(results, query, judgements):
    rel = []
    for r in results:
        key = query + "|||" + r["paper_id"]
        score = judgements.get(key, None)
        if score is not None and score >= 1:
            rel.append(1)
        else:
            rel.append(0)
    return rel


def explanation_quality(bm25_top, sem_top, hyb_top):
    scores = {}

    td = bm25_top.get("term_details", [])
    scores["BM25"] = {
        "score_breakdown": 2 if td else 0,
        "keyword_match": 2 if td else 0,
        "semantic_signal": 0,
        "field_detail": 2 if (any(t["field"] == "title" for t in td) and any(t["field"] == "abstract" for t in td)) else (1 if td else 0),
    }

    matched = sem_top.get("matched_tokens", [])
    scores["Semantic"] = {
        "score_breakdown": 1,
        "keyword_match": 2 if matched else 1,
        "semantic_signal": 2,
        "field_detail": 1,
    }

    td2 = hyb_top.get("term_details", [])
    scores["Hybrid"] = {
        "score_breakdown": 2 if (hyb_top.get("bm25_raw", 0) > 0 and hyb_top.get("ce_raw", 0) != 0) else 1,
        "keyword_match": 2 if hyb_top.get("matched_tokens") else 1,
        "semantic_signal": 2,
        "field_detail": 2 if (any(t["field"] == "title" for t in td2) and any(t["field"] == "abstract" for t in td2)) else 1,
    }

    for method in scores:
        total = 0
        for val in scores[method].values():
            total = total + val
        scores[method]["total"] = total

    return scores


def manual_judging(es, resources, queries, top_k=5):
    judgements = load_judgements()

    print("MANUAL RELEVANCE JUDGING")
    print("rate each paper for the given query:")
    print("2 = highly relevant")
    print("1 = partially relevant")
    print("0 = not relevant")
    print("s = skip")
    print("q = quit and save")
    print("showing top " + str(top_k) + " hybrid results per query.")

    for qi in range(len(queries)):
        query = queries[qi]
        print()
        print("query " + str(qi + 1) + "/" + str(len(queries)) + ': "' + query + '"')

        results = run_hybrid_search(es, resources, query, top_k)
        judged = 0

        for r in results:
            key = query + "|||" + r["paper_id"]
            if key in judgements:
                print("  [already judged: " + str(judgements[key]) + "] " + r["title"][:65])
                judged = judged + 1
                continue

            print()
            print("  title   : " + r["title"])
            print("  abstract: " + r["abstract"])

            while True:
                raw = input("  rate (0/1/2/s/q): ").strip().lower()
                if raw == "q":
                    save_judgements(judgements)
                    print("saved " + str(len(judgements)) + " judgements to " + judgements_file)
                    return judgements
                elif raw == "s":
                    break
                elif raw in ("0", "1", "2"):
                    judgements[key] = int(raw)
                    judged = judged + 1
                    save_judgements(judgements)
                    break
                else:
                    print("  enter 0, 1, 2, s, or q.")

        print("  judged " + str(judged) + " papers for this query.")

    save_judgements(judgements)
    print("done. total judgements saved: " + str(len(judgements)))
    return judgements


def run_evaluation(es, resources, queries, top_k, judgements):
    all_rows = []
    expl_totals = {"BM25": [], "Semantic": [], "Hybrid": []}

    metric_totals = {}
    for m in ("bm25", "sem", "hyb"):
        for metric in ("p", "ndcg"):
            for k in k_values:
                metric_totals[m + "_" + metric + "@" + str(k)] = []

    print("running evaluation on " + str(len(queries)) + " queries (top_k=" + str(top_k) + ")")
    print("=" * 55)

    bm25_results = []
    sem_results = []
    hyb_results = []

    for qi in range(len(queries)):
        query = queries[qi]
        print("[" + str(qi + 1) + "/" + str(len(queries)) + "] " + query + "...", end=" ", flush=True)
        t0 = time.time()

        bm25_results = run_bm25_search(es, query, top_k)
        sem_results = run_semantic_search(resources, query, top_k)
        hyb_results = run_hybrid_search(es, resources, query, top_k)

        bm25_rel = get_manual_relevance(bm25_results, query, judgements)
        sem_rel = get_manual_relevance(sem_results, query, judgements)
        hyb_rel = get_manual_relevance(hyb_results, query, judgements)

        row = {"query": query}
        for k in k_values:
            row["bm25_p@" + str(k)] = round(precision_at_k(bm25_rel, k), 4)
            row["bm25_ndcg@" + str(k)] = round(ndcg_at_k(bm25_rel, k), 4)
            row["sem_p@" + str(k)] = round(precision_at_k(sem_rel, k), 4)
            row["sem_ndcg@" + str(k)] = round(ndcg_at_k(sem_rel, k), 4)
            row["hyb_p@" + str(k)] = round(precision_at_k(hyb_rel, k), 4)
            row["hyb_ndcg@" + str(k)] = round(ndcg_at_k(hyb_rel, k), 4)

        for key in metric_totals:
            metric_totals[key].append(row[key])

        expl = explanation_quality(
            bm25_results[0] if bm25_results else {},
            sem_results[0] if sem_results else {},
            hyb_results[0] if hyb_results else {},
        )
        for method in expl_totals:
            expl_totals[method].append(expl[method]["total"])

        row["expl_BM25"] = expl["BM25"]["total"]
        row["expl_Semantic"] = expl["Semantic"]["total"]
        row["expl_Hybrid"] = expl["Hybrid"]["total"]

        elapsed = time.time() - t0
        print("done (" + str(round(elapsed, 1)) + "s)")

        n_judged = 0
        all_results_for_query = bm25_results + sem_results + hyb_results
        for r in all_results_for_query:
            if query + "|||" + r["paper_id"] in judgements:
                n_judged = n_judged + 1

        print()
        print("  query: " + query)
        print("  papers with manual judgements: " + str(n_judged))
        print()
        print("  metric         BM25    Semantic    Hybrid")
        print("  " + "-" * 42)
        for k in k_values:
            print("  precision@" + str(k) + "    " + str(row["bm25_p@" + str(k)]) + "      " + str(row["sem_p@" + str(k)]) + "      " + str(row["hyb_p@" + str(k)]))
            print("  ndcg@" + str(k) + "         " + str(row["bm25_ndcg@" + str(k)]) + "      " + str(row["sem_ndcg@" + str(k)]) + "      " + str(row["hyb_ndcg@" + str(k)]))
        print()
        print("  explanation quality (out of 8):")
        print("    BM25=" + str(expl["BM25"]["total"]) + "  Semantic=" + str(expl["Semantic"]["total"]) + "  Hybrid=" + str(expl["Hybrid"]["total"]))
        print()

        all_rows.append(row)

    # averages
    avg = {}
    for key in metric_totals:
        vals = metric_totals[key]
        avg[key] = round(sum(vals) / len(vals), 4)

    print("=" * 55)
    print("summary (averaged over all queries)")
    print("=" * 55)
    print()
    print("  metric         BM25    Semantic    Hybrid")
    print("  " + "-" * 42)
    for k in k_values:
        print("  precision@" + str(k) + "    " + str(avg["bm25_p@" + str(k)]) + "      " + str(avg["sem_p@" + str(k)]) + "      " + str(avg["hyb_p@" + str(k)]))
        print("  ndcg@" + str(k) + "         " + str(avg["bm25_ndcg@" + str(k)]) + "      " + str(avg["sem_ndcg@" + str(k)]) + "      " + str(avg["hyb_ndcg@" + str(k)]))

    print()
    print("  explanation quality (avg total score out of 8):")
    for method in ("BM25", "Semantic", "Hybrid"):
        avg_expl = round(sum(expl_totals[method]) / len(expl_totals[method]), 2)
        print("    " + method + ": " + str(avg_expl))

    print()
    print("  explanation quality criteria (scored 0-2 each):")
    print("  criterion               BM25   Semantic   Hybrid   notes")
    print("  " + "-" * 72)
    criteria = {
        "score_breakdown": "does it explain why a paper was ranked?",
        "keyword_match": "does it show which query words matched?",
        "semantic_signal": "does it surface a semantic similarity?",
        "field_detail": "does it show title vs abstract split?",
    }
    sample_expl = explanation_quality(
        bm25_results[0] if bm25_results else {},
        sem_results[0] if sem_results else {},
        hyb_results[0] if hyb_results else {},
    )
    for crit in criteria:
        note = criteria[crit]
        b = sample_expl["BM25"][crit]
        s = sample_expl["Semantic"][crit]
        h = sample_expl["Hybrid"][crit]
        print("  " + crit.ljust(22) + "  " + str(b).rjust(4) + "  " + str(s).rjust(8) + "  " + str(h).rjust(7) + "  " + note)

    print()
    print("  best method per metric:")
    for k in k_values:
        for metric, label in [("p", "precision@" + str(k)), ("ndcg", "ndcg@" + str(k))]:
            scores = {
                "BM25": avg["bm25_" + metric + "@" + str(k)],
                "Semantic": avg["sem_" + metric + "@" + str(k)],
                "Hybrid": avg["hyb_" + metric + "@" + str(k)],
            }
            best = max(scores, key=scores.get)
            print("    " + label.ljust(14) + ": " + best + " (" + str(scores[best]) + ")")

    # save csv
    avg_row = {"query": "AVERAGE"}
    avg_row.update(avg)
    avg_row["expl_BM25"] = round(sum(expl_totals["BM25"]) / len(expl_totals["BM25"]), 2)
    avg_row["expl_Semantic"] = round(sum(expl_totals["Semantic"]) / len(expl_totals["Semantic"]), 2)
    avg_row["expl_Hybrid"] = round(sum(expl_totals["Hybrid"]) / len(expl_totals["Hybrid"]), 2)
    all_rows.append(avg_row)

    f = open("evaluation_results.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    writer.writeheader()
    writer.writerows(all_rows)
    f.close()
    print("results saved to evaluation_results.csv")


if __name__ == "__main__":
    print("arXiv Paper Search System")
    print("Team Members: Mokshad Sankhe (ms5847), Ishant somal(is488) ") 
    print("-" * 35)
    
    print("make sure you are on Drexel VPN! ignore if on Drexel wifi.")
    print("VPN: https://vpn.drexel.edu")
    print()

    es = connect()
    resources = load_resources()

    while True:
        print()
        print("Evaluation Menu")
        print("-" * 35)
        print("1. manual relevance judging")
        print("2. run evaluation")
        print("3. show judgement count")
        print("0. exit")
        print("-" * 35)
        choice = input("select an option: ").strip()

        if choice == "0":
            print("goodbye!")
            break

        elif choice == "3":
            j = load_judgements()
            print("total judgements: " + str(len(j)))
            queries_covered = set()
            for k in j:
                queries_covered.add(k.split("|||")[0])
            for q in sorted(queries_covered):
                count = 0
                for k in j:
                    if k.startswith(q + "|||"):
                        count = count + 1
                print("  " + q + " (" + str(count) + " papers)")

        elif choice == "1":
            n_raw = input("how many queries to judge? (press Enter for all " + str(len(test_queries)) + "): ").strip()
            if n_raw.isdigit():
                n = int(n_raw)
            else:
                n = len(test_queries)
            n = min(n, len(test_queries))
            k_raw = input("how many papers to show per query? (press Enter for 5): ").strip()
            if k_raw.isdigit():
                top_k = int(k_raw)
            else:
                top_k = 5
            manual_judging(es, resources, test_queries[:n], top_k)

        elif choice == "2":
            judgements = load_judgements()
            if not judgements:
                print("no manual judgements found.")
                print("run option 1 first to judge at least some papers.")
                cont = input("continue anyway with all-zero relevance? (yes/no): ").strip().lower()
                if cont != "yes":
                    continue
            n_raw = input("how many queries to evaluate? (press Enter for all " + str(len(test_queries)) + "): ").strip()
            if n_raw.isdigit():
                n = int(n_raw)
            else:
                n = len(test_queries)
            n = min(n, len(test_queries))
            k_raw = input("top-k results per query? (press Enter for 10): ").strip()
            if k_raw.isdigit():
                top_k = int(k_raw)
            else:
                top_k = 10
            run_evaluation(es, resources, test_queries[:n], top_k, judgements)

        else:
            print("invalid choice. please enter 0, 1, 2, or 3.")