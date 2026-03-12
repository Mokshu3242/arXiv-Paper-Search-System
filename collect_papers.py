# collect_papers.py
import requests
import json
import csv
import time
import xml.etree.ElementTree as ET
import os

total_papers = 5000
batch_size = 100
save_folder = "dataset"
max_retries = 5
retry_wait = 10

categories = [
    "cs.AI",
    "cs.LG",
    "cs.CL",
    "cs.CV",
    "cs.NE",
]


def fetch_papers(category, start, max_results):
    url = "http://export.arxiv.org/api/query"
    
    params = {
        "search_query": "cat:" + category,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    
    attempt = 0
    while attempt < max_retries:
        attempt = attempt + 1
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.text
            else:
                print("got a bad status code:", response.status_code, "trying again...")
        except requests.exceptions.Timeout:
            print("request took too long, trying again...")
        except requests.exceptions.ConnectionError:
            print("couldn't connect, trying again...")
        
        time.sleep(retry_wait)
    
    print("gave up after too many failures")
    return None


def parse_xml(xml_text):
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    
    root = ET.fromstring(xml_text)
    papers = []
    
    for entry in root.findall("atom:entry", ns):
        raw_id = entry.find("atom:id", ns).text
        paper_id = raw_id.split("/abs/")[-1]
        paper_id = paper_id.split("v")[0]
        
        title = entry.find("atom:title", ns).text
        title = title.replace("\n", " ").strip()
        
        abstract = entry.find("atom:summary", ns).text
        abstract = abstract.replace("\n", " ").strip()
        
        authors = []
        for author_tag in entry.findall("atom:author", ns):
            name_tag = author_tag.find("atom:name", ns)
            if name_tag is not None:
                authors.append(name_tag.text.strip())
        
        published = entry.find("atom:published", ns).text
        year = int(published[:4])
        
        category_tag = entry.find("arxiv:primary_category", ns)
        if category_tag is not None:
            category = category_tag.attrib.get("term", "unknown")
        else:
            category = "unknown"
        
        paper_url = "https://arxiv.org/abs/" + paper_id
        
        paper = {
            "id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "category": category,
            "url": paper_url,
        }
        
        papers.append(paper)
    
    return papers


def save_json(papers, path):
    f = open(path, "w", encoding="utf-8")
    json.dump(papers, f, indent=2, ensure_ascii=False)
    f.close()
    print("saved json to:", path)


def save_csv(papers, path):
    columns = ["id", "title", "abstract", "authors", "year", "category", "url"]
    
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    
    for paper in papers:
        row = dict(paper)
        row["authors"] = "; ".join(row["authors"])
        writer.writerow(row)
    
    f.close()
    print("saved csv to:", path)


def main():
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    all_papers = []
    seen_ids = set()
    
    papers_per_category = total_papers // len(categories)
    print("i want", total_papers, "papers total")
    print("that means about", papers_per_category, "papers per category")
    print()
    
    for category in categories:
        print("now collecting:", category)
        collected = 0
        start = 0
        
        while collected < papers_per_category:
            remaining = papers_per_category - collected
            batch = min(batch_size, remaining)
            
            print("fetching papers", start, "to", start + batch, "...", end=" ")
            
            xml_text = fetch_papers(category, start, batch)
            
            if xml_text is None:
                print("too many errors, skipping this category")
                break
            
            papers = parse_xml(xml_text)
            
            if len(papers) == 0:
                print("no more papers, moving on")
                break
            
            new_count = 0
            for p in papers:
                if p["id"] not in seen_ids:
                    seen_ids.add(p["id"])
                    all_papers.append(p)
                    new_count = new_count + 1
            
            print("got", len(papers), "papers,", new_count, "were new | total:", len(all_papers))
            
            collected = collected + new_count
            start = start + batch
            
            time.sleep(3)
        
        print()
    
    print("done! collected", len(all_papers), "papers total")
    
    json_path = save_folder + "/papers.json"
    csv_path = save_folder + "/papers.csv"
    
    save_json(all_papers, json_path)
    save_csv(all_papers, csv_path)


main()