# bm25_explain.py
import json
import os
import time
from opensearchpy import OpenSearch, helpers
from dotenv import load_dotenv

load_dotenv()

os_host = os.getenv("OS_HOST", "tux-es2.cci.drexel.edu")
os_port = int(os.getenv("OS_PORT", 9200))
os_user = os.getenv("OS_USER")
os_password = os.getenv("OS_PASSWORD")
index_name = os.getenv("INDEX_NAME", "ms5847_info624_202502_papers")
input_json = "dataset/papers_processed.json"


def connect():
    es = OpenSearch(
        hosts=[{"host": os_host, "port": os_port}],
        http_auth=(os_user, os_password),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )
    
    if not es.ping():
        print("could not connect. make sure you are on Drexel VPN!")
        raise ConnectionError("Cannot reach OpenSearch.")
    
    print("connected to OpenSearch (" + os_host + ")")
    print()
    return es


def check_total_entries(es):
    print()
    print("Check Total Entries:")
    
    if not es.indices.exists(index=index_name):
        print("index does not exist yet. use option 4 to upload data first.")
        return
    
    result = es.count(index=index_name)
    total = result["count"]
    
    print("index:", index_name)
    print("total:", total, "documents")
    
    breakdown = es.search(
        index=index_name,
        body={
            "size": 0,
            "aggs": {
                "by_category": {
                    "terms": {"field": "category", "size": 20}
                }
            }
        }
    )
    
    buckets = breakdown["aggregations"]["by_category"]["buckets"]
    if buckets:
        print()
        print("breakdown by category:")
        for bucket in buckets:
            print(" ", bucket["key"], "->", bucket["doc_count"], "papers")
    
    print()


def delete_single_entry(es):
    print()
    print("Delete Single Entry:")
    print("you need the arxiv paper ID (e.g. 2302.13971)")
    print("type 'cancel' to go back.")
    print()
    
    paper_id = input("enter paper ID to delete: ").strip()
    
    if paper_id.lower() == "cancel":
        print("cancelled.")
        return
    
    if not paper_id:
        print("no ID entered. cancelled.")
        return
    
    if not es.exists(index=index_name, id=paper_id):
        print("paper ID", paper_id, "was not found. nothing was deleted.")
        return
    
    doc = es.get(index=index_name, id=paper_id)
    title = doc["_source"].get("title", "Unknown title")
    
    print()
    print("found paper:")
    print("  id    :", paper_id)
    print("  title :", title[:70])
    print()
    
    confirm = input("are you sure you want to delete this? (yes / no): ").strip().lower()
    
    if confirm == "yes":
        es.delete(index=index_name, id=paper_id)
        print("paper", paper_id, "deleted successfully.")
    else:
        print("cancelled. nothing was deleted.")
    
    print()


def delete_entire_index(es):
    print()
    print("Delete Entire Index:")
    print("WARNING: this will permanently delete ALL documents in:")
    print(" ", index_name)
    print()
    
    if not es.indices.exists(index=index_name):
        print("index does not exist. nothing to delete.")
        return
    
    total = es.count(index=index_name)["count"]
    print("this index currently has", total, "documents.")
    print()
    
    confirm1 = input("type DELETE to confirm: ").strip()
    if confirm1 != "DELETE":
        print("cancelled. index was NOT deleted.")
        return
    
    confirm2 = input("are you really sure? (yes / no): ").strip().lower()
    if confirm2 != "yes":
        print("cancelled. index was NOT deleted.")
        return
    
    es.indices.delete(index=index_name)
    print("index", index_name, "has been deleted.")
    print()


def upload_data(es):
    print()
    print("Upload Data From File:")
    print("file:", input_json)
    print()
    
    if not os.path.exists(input_json):
        print("file not found:", input_json)
        print("run collect_papers.py and preprocess_papers.py first.")
        return
    
    f = open(input_json, "r", encoding="utf-8")
    papers = json.load(f)
    f.close()
    
    print("found", len(papers), "papers in file.")
    
    if not es.indices.exists(index=index_name):
        print("index does not exist, creating it now...")
        create_index(es)
    else:
        existing = es.count(index=index_name)["count"]
        print("index already exists with", existing, "documents.")
        choice = input("do you want to (a) add to existing  or  (b) wipe and re-upload? (a/b): ").strip().lower()
        if choice == "b":
            es.indices.delete(index=index_name)
            print("old index deleted.")
            create_index(es)
    
    actions = []
    for paper in papers:
        if isinstance(paper.get("authors"), list):
            authors_str = " ".join(paper["authors"])
        else:
            authors_str = paper.get("authors", "")
        
        actions.append({
            "_index": index_name,
            "_id": paper["id"],
            "_source": {
                "paper_id": paper["id"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "authors": authors_str,
                "year": paper.get("year", 0),
                "category": paper.get("category", "unknown"),
                "url": paper.get("url", ""),
            }
        })
    
    print("uploading", len(actions), "papers...")
    ok, failed = helpers.bulk(es, actions, raise_on_error=False, stats_only=True)
    time.sleep(1)
    
    print("uploaded:", ok)
    print("failed  :", failed)
    
    total = es.count(index=index_name)["count"]
    print("index now has", total, "documents total.")
    print()


def create_index(es):
    index_config = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "similarity": {
                "custom_bm25": {"type": "BM25", "k1": 1.2, "b": 0.75}
            }
        },
        "mappings": {
            "properties": {
                "title":    {"type": "text",    "similarity": "custom_bm25", "analyzer": "english"},
                "abstract": {"type": "text",    "similarity": "custom_bm25", "analyzer": "english"},
                "authors":  {"type": "text",    "similarity": "custom_bm25"},
                "year":     {"type": "integer"},
                "category": {"type": "keyword"},
                "paper_id": {"type": "keyword"},
                "url":      {"type": "keyword"},
            }
        }
    }
    es.indices.create(index=index_name, body=index_config)
    print("index created with BM25.")


def explain_search(es):
    print()
    print("Explain BM25 Search:")
    print("this shows you how each result got its score, including TF, IDF, and document length normalization.")
    print("type 'cancel' to go back.")
    print()
    
    query = input("enter your search query: ").strip()
    
    if query.lower() == "cancel" or not query:
        print("cancelled.")
        return
    
    top_k_input = input("how many results to show? (press Enter for 5): ").strip()
    if top_k_input.isdigit():
        top_k = int(top_k_input)
    else:
        top_k = 5
    
    print()
    print("searching for:", query)
    print("explain=True is ON, OpenSearch will return the full scoring breakdown")
    print()
    
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
    
    hits = response["hits"]["hits"]
    total = response["hits"]["total"]["value"]
    
    print("found", total, "matching papers. showing top", len(hits), "with full explanation.")
    print()
    
    if not hits:
        print("no results found.")
        return
    
    all_scores = []
    for h in hits:
        all_scores.append(round(h["_score"], 4))
    has_ties = len(all_scores) != len(set(all_scores))
    
    for rank in range(len(hits)):
        print_one_explanation(hits[rank], rank + 1)
    
    if has_ties:
        print("NOTE: some papers have the same score (a tie).")
        print()
        print("why does this happen?")
        print("BM25 score = IDF(word) x ( TF x (k1+1) ) / ( TF + k1 x (1 - b + b x docLen/avgDocLen) )")
        print("when multiple papers use the same query words the same number of times in similarly-sized text, they get identical scores.")
        print("this is normal in keyword search.")
        print()
        print("real search engines break ties using extra signals like:")
        print("- publication date (newer papers ranked higher)")
        print("- citation count (popular papers ranked higher)")
        print("- click data (user engagement signals)")
        print("your BM25 system is still correct, ties just mean those papers are equally relevant by keyword matching.")
    
    print("SUMMARY TABLE:")
    print("Rank   Total Score   Title Score   Abstract Score   Title")
    
    for rank in range(len(hits)):
        hit = hits[rank]
        total_score = round(hit["_score"], 4)
        term_details = get_term_details(hit.get("_explanation", {}))
        
        title_score = 0.0
        abstract_score = 0.0
        for t in term_details:
            if t["field"] == "title":
                title_score = title_score + t["final"]
            if t["field"] == "abstract":
                abstract_score = abstract_score + t["final"]
        
        title_score = round(title_score, 4)
        abstract_score = round(abstract_score, 4)
        short_title = hit["_source"]["title"][:35]
        
        print("#" + str(rank + 1), " ", total_score, " ", title_score, " ", abstract_score, " ", short_title)
    
    print()
    print("Legend:")
    print("Total Score = final BM25 score (sum of all matched term scores, with field boosts applied)")
    print("Title Score = sum of per-term BM25 scores matched in the title field (3x boost included)")
    print("Abstract Score = sum of per-term BM25 scores matched in the abstract field (2x boost included)")


def print_one_explanation(hit, rank):
    paper = hit["_source"]
    total_score = round(hit["_score"], 4)
    explanation = hit.get("_explanation", {})
    
    print("Rank #" + str(rank) + " Total BM25 Score =", total_score)
    print("title :", paper["title"][:70])
    print("year :", paper["year"], "  category:", paper["category"])
    print("url :", paper["url"])
    print()
    print("BM25 Formula (how OpenSearch calculated the score):")
    print("score = ΣIDF(word) x ( TF x (k1+1) ) / ( TF + k1 x (1 - b + b x docLen/avgDocLen) )")
    print("where: k1=1.2  b=0.75")
    print("each matching word contributes one term to the sum.")
    print()
    print("Per-Word Breakdown:")
    print("(each query word that matched contributes a separate score)")
    print()
    
    term_details = get_term_details(explanation)
    
    if term_details:
        print("  Word      Field        TF Score      IDF       Final")
        print("  " + "-" * 58)
        for t in term_details:
            print("  " + t["term"].ljust(18), t["field"].ljust(10), str(t["tf_score"]).rjust(10), str(t["idf"]).rjust(8), str(t["final"]).rjust(10))
    else:
        print("(term-level breakdown not available for this result)")
    
    title_score = 0.0
    abstract_score = 0.0
    for t in term_details:
        if t["field"] == "title":
            title_score = title_score + t["final"]
        if t["field"] == "abstract":
            abstract_score = abstract_score + t["final"]
    
    title_score = round(title_score, 4)
    abstract_score = round(abstract_score, 4)
    
    print()
    print("Field Score Totals:")
    if title_score > 0 or abstract_score > 0:
        print("title match   :", title_score, "(sum of all matched-term scores in title, 3x boost included)")
        print("abstract match:", abstract_score, "(sum of all matched-term scores in abstract, 2x boost included)")
        print("total:", total_score)
        print()
        print("note: each field is searched independently with its boost applied (title^3, abstract^2).")
        print("the per-term scores shown above already include those boosts.")
        print("the total score is the sum of all matched term contributions across both fields.")
    else:
        print("total:", total_score)
        print("(field breakdown not available)")
    
    print()
    print("Query Word Matches:")
    print("(terms shown are the analyzed/stemmed forms OpenSearch actually matched)")
    
    matched_in_title = []
    matched_in_abstract = []
    for t in term_details:
        if t["field"] == "title":
            matched_in_title.append(t["term"])
        if t["field"] == "abstract":
            matched_in_abstract.append(t["term"])
    
    if matched_in_title:
        print("matched in title:")
        for w in matched_in_title:
            print(" ", w)
    else:
        print("matched in title: (none)")
    
    if matched_in_abstract:
        print("matched in abstract:")
        for w in matched_in_abstract:
            print(" ", w)
    else:
        print("matched in abstract: (none)")
    
    print()


def get_term_details(explanation):
    term_details = []
    
    def walk(node, current_field=None, current_term=None):
        if node is None:
            return
        
        description = node.get("description", "")
        value = node.get("value", 0.0)
        children = node.get("details", [])
        
        if "weight(" in description:
            if "title:" in description:
                field = "title"
                term = description.split("title:")[1].split(" ")[0]
            elif "abstract:" in description:
                field = "abstract"
                term = description.split("abstract:")[1].split(" ")[0]
            else:
                field = "other"
                term = "?"
            
            idf_val = 0.0
            tf_val = 0.0
            
            def find_tf_idf(n):
                nonlocal idf_val, tf_val
                if n is None:
                    return
                desc = n.get("description", "")
                val = n.get("value", 0.0)
                if desc.startswith("idf"):
                    idf_val = round(val, 4)
                if desc.startswith("tf,") or desc.startswith("tf "):
                    tf_val = round(val, 4)
                for c in n.get("details", []):
                    find_tf_idf(c)
            
            for child in children:
                find_tf_idf(child)
            
            term_details.append({
                "term": term,
                "field": field,
                "idf": idf_val,
                "tf_score": tf_val,
                "final": round(value, 4),
            })
        
        for child in children:
            walk(child)
    
    walk(explanation)
    return term_details


def get_idf_values(explanation):
    idf_values = {}
    
    def walk(node):
        if node is None:
            return
        
        description = node.get("description", "")
        value = node.get("value", 0.0)
        children = node.get("details", [])
        
        if description.startswith("idf") and "term" in description:
            try:
                word = description.split("'")[1]
                idf_values[word] = round(value, 4)
            except IndexError:
                pass
        
        for child in children:
            walk(child)
    
    walk(explanation)
    return idf_values


def show_menu():
    print("arXiv Paper Search System")
    print("Team Members: Mokshad Sankhe (ms5847), Ishant somal(is488) ") 
    print("-" * 35)
    print()
    print("OpenSearch Manager: arXiv Papers")
    print("1. check total entries")
    print("2. delete a single entry (by paper ID)")
    print("3. delete entire index (irreversible!)")
    print("4. upload data from file")
    print("5. search with score explanation")
    print("6. exit")

if __name__ == "__main__":
    print("NOTE: make sure you are on Drexel VPN!")
    print("VPN: https://vpn.drexel.edu")
    print()
    es = connect()
    while True:
        show_menu()
        choice = input("enter your choice (1-6): ").strip()
        if choice == "1":
            check_total_entries(es)
        elif choice == "2":
            delete_single_entry(es)
        elif choice == "3":
            delete_entire_index(es)
        elif choice == "4":
            upload_data(es)
        elif choice == "5":
            explain_search(es)
        elif choice == "6":
            print("goodbye!")
            break
        else:
            print("invalid choice. please enter a number between 1 and 6.")