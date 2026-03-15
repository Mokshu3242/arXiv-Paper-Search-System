# hybrid.py
import time
import numpy as np
from opensearchpy import OpenSearch
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity

from bm25_explain import connect, get_term_details, index_name
from bert_explain import load_resources, preprocess_query, score_label

load_dotenv()

alpha = 0.4
beta = 0.6

bm25_candidates = 50
semantic_candidates = 50


def normalise(scores):
    lo = scores.min()
    hi = scores.max()
    if hi == lo:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def bm25_search(es, query, top_k):
    response = es.search(
        index=index_name,
        explain=True,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "abstract^2"],
                    "type": "best_fields",
                }
            },
            "size": top_k,
            "_source": ["paper_id", "title", "abstract", "year", "category", "url"],
        }
    )
    hits = []
    for h in response["hits"]["hits"]:
        hits.append({
            "paper_id": h["_source"]["paper_id"],
            "bm25_raw": h["_score"],
            "explanation": h.get("_explanation", {}),
            "source": h["_source"],
        })
    return hits


def get_bert_candidates(query, top_k, abstracts, titles, bi_encoder, abs_embeddings, title_embeddings):
    q_emb = bi_encoder.encode(query, convert_to_numpy=True)
    abs_sim = cosine_similarity([q_emb], abs_embeddings)[0]
    title_sim = cosine_similarity([q_emb], title_embeddings)[0]
    combined = 0.6 * abs_sim + 0.4 * title_sim
    top_idx = np.argsort(combined)[::-1][:top_k]
    results = []
    for i in top_idx:
        results.append({
            "idx": i,
            "bert_bi": float(combined[i]),
            "abs_sim": float(abs_sim[i]),
            "title_sim": float(title_sim[i]),
        })
    return results


def hybrid_search(es, query, top_k, abstracts, titles, urls, all_tokens, paper_ids, bi_encoder, cross_encoder, abs_embeddings, title_embeddings):
    query_tokens = preprocess_query(query)

    print("[1/3] BM25 retrieval (top " + str(bm25_candidates) + ")...")
    bm25_hits = bm25_search(es, query, bm25_candidates)
    bm25_by_id = {}
    for h in bm25_hits:
        bm25_by_id[h["paper_id"]] = h

    print("[2/3] semantic retrieval (top " + str(semantic_candidates) + ")...")
    bert_hits = get_bert_candidates(query, semantic_candidates, abstracts, titles, bi_encoder, abs_embeddings, title_embeddings)
    bert_by_idx = {}
    for h in bert_hits:
        bert_by_idx[h["idx"]] = h

    pid_to_idx = {}
    for i in range(len(paper_ids)):
        pid_to_idx[paper_ids[i]] = i

    candidate_indices = set()
    for h in bm25_hits:
        idx = pid_to_idx.get(h["paper_id"])
        if idx is not None:
            candidate_indices.add(idx)
    for h in bert_hits:
        candidate_indices.add(h["idx"])
    candidate_indices = list(candidate_indices)

    print("[3/3] cross-encoder reranking...")
    ce_inputs = []
    for i in candidate_indices:
        ce_inputs.append([query, titles[i] + " " + abstracts[i][:512]])
    ce_scores = cross_encoder.predict(ce_inputs, show_progress_bar=False)

    bm25_raw_arr = np.array([bm25_by_id.get(paper_ids[i], {}).get("bm25_raw", 0.0) for i in candidate_indices])
    ce_raw_arr = np.array(ce_scores)
    bm25_norm = normalise(bm25_raw_arr)
    ce_norm = normalise(ce_raw_arr)
    final_arr = alpha * bm25_norm + beta * ce_norm
    order = np.argsort(final_arr)[::-1][:top_k]

    results = []
    for rank_pos in order:
        i = candidate_indices[rank_pos]
        pid = paper_ids[i]
        matched = query_tokens & all_tokens[i]
        bm25_info = bm25_by_id.get(pid, {})
        term_dets = get_term_details(bm25_info.get("explanation", {}))
        title_bm25 = 0.0
        abs_bm25 = 0.0
        for t in term_dets:
            if t["field"] == "title":
                title_bm25 = title_bm25 + t["final"]
            if t["field"] == "abstract":
                abs_bm25 = abs_bm25 + t["final"]
        bert_info = bert_by_idx.get(i, {})
        results.append({
            "rank": len(results) + 1,
            "paper_id": pid,
            "title": titles[i],
            "abstract": abstracts[i],
            "url": urls[i],
            "bm25_raw": round(float(bm25_raw_arr[rank_pos]), 4),
            "ce_raw": round(float(ce_raw_arr[rank_pos]), 4),
            "bm25_norm": round(float(bm25_norm[rank_pos]), 4),
            "ce_norm": round(float(ce_norm[rank_pos]), 4),
            "final_score": round(float(final_arr[rank_pos]), 4),
            "bm25_title": round(title_bm25, 4),
            "bm25_abstract": round(abs_bm25, 4),
            "term_details": term_dets,
            "bert_abs_sim": round(bert_info.get("abs_sim", 0.0), 4),
            "bert_title_sim": round(bert_info.get("title_sim", 0.0), 4),
            "bert_bi": round(bert_info.get("bert_bi", 0.0), 4),
            "matched_tokens": sorted(matched),
            "in_bm25_pool": pid in bm25_by_id,
            "in_bert_pool": i in bert_by_idx,
        })
    return results


def print_result(r):
    bm25_contrib = round(alpha * r["bm25_norm"], 4)
    ce_contrib = round(beta * r["ce_norm"], 4)
    total = r["final_score"]
    bm25_pct = round((bm25_contrib / total * 100) if total else 0, 1)
    ce_pct = round((ce_contrib / total * 100) if total else 0, 1)
    bar_bm25 = "█" * int(bm25_pct / 5)
    bar_ce = "█" * int(ce_pct / 5)

    pool_tags = []
    if r["in_bm25_pool"]:
        pool_tags.append("BM25")
    if r["in_bert_pool"]:
        pool_tags.append("Semantic")

    print("-----")
    print("rank: #" + str(r["rank"]))
    print("title: " + r["title"])
    print("url: " + r["url"])
    print("found in: " + " + ".join(pool_tags))
    print()
    print("final score: " + str(total) + "  " + score_label(r["ce_raw"]))
    print("formula: final = alpha x bm25_norm + beta x ce_norm")
    print("       = " + str(alpha) + " x " + str(r["bm25_norm"]) + " + " + str(beta) + " x " + str(r["ce_norm"]))
    print("       = " + str(bm25_contrib) + " + " + str(ce_contrib) + " = " + str(total))
    print()
    print("bm25 part: " + str(bm25_contrib) + " (" + str(bm25_pct) + "%) " + bar_bm25)
    print("bert part: " + str(ce_contrib) + " (" + str(ce_pct) + "%) " + bar_ce)
    print()
    print("bm25 scores:")
    print("  raw: " + str(r["bm25_raw"]))
    print("  normalised: " + str(r["bm25_norm"]))
    print("  title field total (3x boost): " + str(r["bm25_title"]))
    print("  abstract field total (2x boost): " + str(r["bm25_abstract"]))
    if r["term_details"]:
        print("  word breakdown:")
        print("  word               field       tf score    idf       final")
        print("  " + "-" * 56)
        for t in r["term_details"]:
            print("  " + t["term"].ljust(18), t["field"].ljust(10), str(t["tf_score"]).rjust(10), str(t["idf"]).rjust(8), str(t["final"]).rjust(10))
    elif r["in_bm25_pool"]:
        print("  (word breakdown not available)")
    else:
        print("  (this paper was not in the bm25 pool, score = 0)")
    print()
    print("bert scores:")
    print("  cross-encoder raw: " + str(r["ce_raw"]))
    print("  cross-encoder normalised: " + str(r["ce_norm"]))
    print("  bi-encoder combined: " + str(r["bert_bi"]) + " (0.6 x abstract + 0.4 x title)")
    print("  abstract cosine sim: " + str(r["bert_abs_sim"]))
    print("  title cosine sim: " + str(r["bert_title_sim"]))
    print()
    print("keyword matches:")
    if r["matched_tokens"]:
        print("  matched: " + ", ".join(r["matched_tokens"]))
    else:
        print("  no keyword overlap, ranked by semantic similarity only")
    print("abstract preview: " + r["abstract"][:200] + "...")


def print_summary_table(results):
    print()
    print("summary table:")
    print("rank  final     bm25      bm25%   bert      bert%   ce raw   found in         title")
    print("-" * 95)
    for r in results:
        bm25_c = round(alpha * r["bm25_norm"], 4)
        ce_c = round(beta * r["ce_norm"], 4)
        total = r["final_score"]
        bp = round((bm25_c / total * 100) if total else 0, 1)
        cp = round((ce_c / total * 100) if total else 0, 1)
        srcs = ""
        if r["in_bm25_pool"]:
            srcs = srcs + "BM25"
        if r["in_bert_pool"]:
            if srcs:
                srcs = srcs + "+Sem."
            else:
                srcs = "Sem."
        print(
            "#" + str(r["rank"]).ljust(4),
            str(total).rjust(7),
            str(bm25_c).rjust(8),
            (str(bp) + "%").rjust(7),
            str(ce_c).rjust(8),
            (str(cp) + "%").rjust(6),
            str(r["ce_raw"]).rjust(8),
            srcs.ljust(16),
            r["title"][:35]
        )
    print()
    print("legend:")
    print("final = alpha(" + str(alpha) + ") x bm25_norm + beta(" + str(beta) + ") x ce_norm")
    print("bm25  = how much bm25 keyword matching contributed")
    print("bert  = how much the cross-encoder semantic score contributed")
    print("ce raw = raw cross-encoder score before normalising")
    print("found in = which search pool found this paper")


if __name__ == "__main__":
    print("arXiv Paper Search System")
    print("Team Members: Mokshad Sankhe (ms5847), Ishant somal(is488) ") 
    print("-" * 35)
    print("make sure you are on Drexel VPN! ignore if on Drexel wifi.")
    print("VPN: https://vpn.drexel.edu")

    es = connect()
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

    while True:
        print()
        print("Hybrid BM25 + BERT Search (alpha=" + str(alpha) + ", beta=" + str(beta) + ")")
        print("=" * 60)

        query = input("search query (or 'exit' to quit): ").strip()
        if query.lower() in ("exit", "quit", "q"):
            print("goodbye!")
            break
        if not query:
            continue

        top_k_raw = input("how many results? (press Enter for 5): ").strip()
        if top_k_raw.isdigit():
            top_k = int(top_k_raw)
        else:
            top_k = 5

        alpha_raw = input("alpha - bm25 weight 0 to 1 (press Enter for " + str(alpha) + "): ").strip()
        beta_raw = input("beta  - bert weight 0 to 1 (press Enter for " + str(beta) + "): ").strip()

        try:
            alpha = float(alpha_raw)
        except ValueError:
            alpha = 0.4
        try:
            beta = float(beta_raw)
        except ValueError:
            beta = 0.6

        if alpha < 0.0:
            alpha = 0.0
        if alpha > 1.0:
            alpha = 1.0
        if beta < 0.0:
            beta = 0.0
        if beta > 1.0:
            beta = 1.0

        if abs(alpha + beta - 1.0) > 0.01:
            print("note: alpha + beta = " + str(round(alpha + beta, 2)) + " (not 1.0). scores still work but wont be in the 0 to 1 range.")

        print()
        print("searching for: '" + query + "'  [alpha=" + str(alpha) + ", beta=" + str(beta) + "]")
        t0 = time.time()
        results = hybrid_search(
            es, query, top_k,
            abstracts, titles, urls, all_tokens, paper_ids,
            bi_encoder, cross_encoder,
            abs_embeddings, title_embeddings
        )
        elapsed = time.time() - t0
        print("done in " + str(round(elapsed, 2)) + " seconds")

        if not results:
            print("no results found.")
            continue

        for r in results:
            print_result(r)

        print_summary_table(results)