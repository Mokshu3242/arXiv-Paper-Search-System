# main.py
import os
import sys
import subprocess


def run_script(script_file):
    if not os.path.exists(script_file):
        print("file not found: " + script_file)
        print("make sure all scripts are in the same directory as main.py.")
        return

    print("launching " + script_file + "...")
    print()
    subprocess.run([sys.executable, script_file])
    print()
    print(script_file + " finished.")


def dataset_status():
    files = {
        "dataset/papers.json": "raw papers (JSON)",
        "dataset/papers.csv": "raw papers (CSV)",
        "dataset/papers_processed.json": "processed papers (JSON)",
        "dataset/papers_processed.csv": "processed papers (CSV)",
        "dataset/paper_embeddings.npy": "abstract embeddings (.npy)",
        "dataset/title_embeddings.npy": "title embeddings (.npy)",
    }

    print()
    print("dataset status:")
    print("-" * 50)
    for path in files:
        label = files[path]
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            if size_kb < 1024:
                size_str = str(round(size_kb)) + " KB"
            else:
                size_str = str(round(size_kb / 1024, 1)) + " MB"
            print("  [found]   " + label + " (" + size_str + ")")
        else:
            print("  [missing] " + label)
    print()


def show_menu():
    print()
    print("arXiv Paper Search System")
    print("Team Members: Mokshad Sankhe (ms5847), Ishant somal(is488) ") 
    print("-" * 35)
    print("1. collect papers")
    print("2. preprocess papers")
    print("3. BM25 search (OpenSearch)")
    print("4. BERT semantic search")
    print("5. hybrid search (BM25 + BERT)")
    print("6. evaluate (Precision@K, nDCG, comparison)")
    print("7. check dataset status")
    print("0. exit")
    print("-" * 35)


if __name__ == "__main__":
    while True:
        show_menu()
        choice = input("select an option: ").strip()

        if choice == "1":
            run_script("collect_papers.py")
        elif choice == "2":
            run_script("preprocess_papers.py")
        elif choice == "3":
            run_script("bm25_explain.py")
        elif choice == "4":
            run_script("bert_explain.py")
        elif choice == "5":
            run_script("hybrid.py")
        elif choice == "6":
            run_script("evaluate.py")
        elif choice == "7":
            dataset_status()
        elif choice == "0":
            print("goodbye!")
            sys.exit(0)
        else:
            print("invalid choice. please enter a number between 0 and 7.")