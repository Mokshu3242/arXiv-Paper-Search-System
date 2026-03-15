# arXiv Paper Search System

An explainable search engine for academic papers that shows you exactly why each result was ranked the way it was. Combines traditional BM25 keyword search with BERT semantic search and a hybrid approach that uses both.

---

## What each file does

| File | What it does |
|------|-------------|
| `collect_papers.py` | Downloads 5000 papers from arXiv API |
| `preprocess_papers.py` | Cleans and tokenizes the paper text |
| `bm25_explain.py` | BM25 keyword search via OpenSearch, with full score explanation |
| `bert_explain.py` | BERT semantic search using sentence transformers |
| `hybrid.py` | Combines BM25 + BERT with alpha/beta weights |
| `evaluate.py` | Runs Precision@K and nDCG evaluation across all 3 methods |
| `main.py` | Menu launcher for all of the above |

---

## Requirements

- Python 3.9 or higher
- Access to Drexel VPN (for OpenSearch connection)
- Internet connection (for downloading models and papers)

---

## Setup

### 1. Clone or download the project

Put all the `.py` files in the same folder.

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Create a `.env` file

Create a file called `.env` in the same folder as the scripts and add your OpenSearch credentials:

```
OS_HOST=tux-es2.cci.drexel.edu
OS_PORT=9200
OS_USER=your_username_here
OS_PASSWORD=your_password_here
INDEX_NAME=your_index_name_here
```

Replace `your_username_here`, `your_password_here`, and `your_index_name_here` with your actual credentials.

### 4. Connect to Drexel VPN

Before running anything that uses OpenSearch (BM25 search, hybrid search, evaluate), make sure you are connected to the Drexel VPN.

VPN link: https://vpn.drexel.edu

If you are already on Drexel Wi-Fi you can skip this.

---

## How to run

The easiest way is to use the main launcher:

```
python main.py
```

This will show a menu:

```
arXiv Paper Search System
-----------------------------------
1. collect papers
2. preprocess papers
3. BM25 search (OpenSearch)
4. BERT semantic search
5. hybrid search (BM25 + BERT)
6. evaluate (Precision@K, nDCG, comparison)
7. check dataset status
0. exit
```

---

## Step by step first time setup

Follow these steps in order the first time you run the project.

### Step 1 - Collect papers

Run option 1 from main menu, or:

```
python collect_papers.py
```

This downloads ~5000 papers from arXiv and saves them to:
- `dataset/papers.json`
- `dataset/papers.csv`

Takes around 10-15 minutes because of API rate limits.

### Step 2 - Preprocess papers

Run option 2 from main menu, or:

```
python preprocess_papers.py
```

This cleans the text (lowercase, remove stopwords, lemmatize) and saves to:
- `dataset/papers_processed.json`
- `dataset/papers_processed.csv`

### Step 3 - Upload papers to OpenSearch

Run option 3 from main menu, then choose option 4 (upload data from file) inside the BM25 menu.

Make sure you are on Drexel VPN before this step.

This creates the OpenSearch index and uploads all 5000 papers.

### Step 4 - Run a search

You can now run any of the three search options:

**BM25 search** - option 3 from main menu, then option 5 (search with score explanation)

**BERT search** - option 4 from main menu

**Hybrid search** - option 5 from main menu

The first time you run BERT or hybrid search it will compute embeddings for all 5000 papers and save them to:
- `dataset/paper_embeddings.npy`
- `dataset/title_embeddings.npy`

This takes a few minutes but only happens once. After that it loads them from disk instantly.

---

## Running the evaluation

Run option 6 from the main menu, or:

```
python evaluate.py
```

The evaluation menu has 3 options:

### Option 1 - Manual relevance judging

You need to do this before running the evaluation. It shows you papers one by one and asks you to rate them:

- `2` = highly relevant to the query
- `1` = partially relevant
- `0` = not relevant
- `s` = skip this paper
- `q` = quit and save progress

Ratings are saved to `manual_judgements.json` automatically after each rating. You can stop and continue later without losing progress.

**Tip:** Set papers per query to at least 10 so that Precision@10 and nDCG@10 are meaningful.

### Option 2 - Run evaluation

Runs all 3 search methods on all 25 test queries and computes:
- Precision@1, @3, @5, @10
- nDCG@1, @3, @5, @10
- Explanation quality score for each method

Results are saved to `evaluation_results.csv`.

### Option 3 - Show judgement count

Shows how many papers you have rated so far, broken down by query.

---

## Dataset folder structure

After running everything, your `dataset/` folder should look like this:

```
dataset/
    papers.json                  <- raw papers from arXiv
    papers.csv                   <- same data in CSV format
    papers_processed.json        <- cleaned and tokenized papers
    papers_processed.csv         <- same data in CSV format
    paper_embeddings.npy         <- BERT abstract embeddings (auto-generated)
    title_embeddings.npy         <- BERT title embeddings (auto-generated)
```

---

## Common issues

**"could not connect" error**
Make sure you are on Drexel VPN. If you are on Drexel Wi-Fi you should be fine without VPN.

**"file not found: dataset/papers_processed.json"**
You need to run collect_papers.py and preprocess_papers.py first before running any search.

**"index does not exist"**
You need to upload the data first. Run bm25_explain.py and choose option 4 (upload data from file).

**Embeddings taking a long time**
The first time you run BERT or hybrid search it needs to compute embeddings for all 5000 papers. This is normal and only happens once. After that it loads from the `.npy` files instantly.

**UNEXPECTED key warnings from sentence-transformers**
These warnings about `embeddings.position_ids` can be safely ignored. They do not affect the search results.

---

## Team Members
Mokshad Sankhe (ms5847), Ishant somal(is488)