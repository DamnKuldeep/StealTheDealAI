import sys
from pathlib import Path

# torch must be imported before chromadb - see the note in agents/frontier_agent.py.
# chromadb drags in onnxruntime's native libraries, and on Windows loading those first
# makes the later torch import abort the process outright, with no traceback.
import torch  # isort: skip
import chromadb  # isort: skip
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

RAG_CATALOG = settings.DATA_PROCESSED_DIR / "rag_catalog.csv"
DB_PATH = settings.VECTORSTORE_DIR / "products_vectorstore"

# Bigger batches keep the GPU fed; all-MiniLM-L6-v2 is small enough that even a
# 4GB laptop GPU has plenty of headroom at this batch size.
ENCODE_BATCH_SIZE = 256


def build():
    if not RAG_CATALOG.exists():
        print(f"File not found: {RAG_CATALOG}")
        print("Please run clean_rag_data.py first.")
        return

    print("Loading RAG catalog...")
    df = pd.read_csv(RAG_CATALOG)
    print(f"Loaded {len(df)} products.")

    # How many catalog rows to embed - tune via notebooks/01_data_exploration.ipynb
    # (it benchmarks embedding throughput on this machine) then set
    # settings.MAX_RAG_ITEMS accordingly.
    max_items = min(settings.MAX_RAG_ITEMS, len(df))
    df = df.head(max_items)
    print(f"Building vector store for {max_items} products...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model onto {device}...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    print("Connecting to ChromaDB...")
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection("products")

    BATCH_SIZE = 1000
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i:i + BATCH_SIZE]

        documents = batch['document_text'].tolist()
        ids = [str(idx) for idx in batch.index.tolist()]
        metadatas = [
            {
                "name": str(row['name']),
                "category": str(row['category']),
                "price": float(row['price']),
            }
            for _, row in batch.iterrows()
        ]

        embeddings = model.encode(
            documents, batch_size=ENCODE_BATCH_SIZE, show_progress_bar=False
        ).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    print(f"Successfully built vector store at {DB_PATH}")
    print(f"Total items in collection: {collection.count()}")

if __name__ == "__main__":
    build()
