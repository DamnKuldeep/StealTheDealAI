import threading
from typing import Optional

# Import order matters and is NOT alphabetical on purpose: torch must be imported
# before chromadb. chromadb pulls in onnxruntime and other native libraries, and on
# Windows loading those first makes the subsequent torch import abort the whole
# process - a hard crash with no Python traceback and no output at all, which is
# maddening to diagnose. Keep torch first.
import torch  # isort: skip
import chromadb  # isort: skip
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from agents.deals import Deal
from agents.items import Item
from agents.preprocessor import Preprocessor
from agents.base_agent import Agent
from agents.rate_limiter import call_with_model_fallback
from config import settings


class FrontierAgent(Agent):
    """
    RAG Agent: Queries a ChromaDB of ~400k products to find similar items
    and uses an LLM to estimate the price.
    """
    SYSTEM_PROMPT = """You estimate the price of a product based on its description and a list of similar products. 
    Respond strictly with the estimated price as a single float number in INR (₹). Do not include any other text, symbols, or commas."""

    name = "RAG Agent"
    color = Agent.YELLOW

    def __init__(self, db_path: str = None):
        self.log("RAG Agent initializing with SentenceTransformer and ChromaDB")
        self.preprocessor = Preprocessor()

        self.models = settings.LLM_MODELS
        if settings.LLM_PROVIDER == "nim":
            self.client = OpenAI(
                base_url=settings.LLM_BASE_URL,
                api_key=settings.LLM_API_KEY,
                timeout=settings.NIM_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )
        else:
            self.client = OpenAI()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        db_path = db_path or str(settings.VECTORSTORE_DIR / "products_vectorstore")
        self.db = chromadb.PersistentClient(path=db_path)
        self.collection = self.db.get_or_create_collection("products")
        # PlanningAgent evaluates MAX_CONCURRENT_DEALS deals at once, and each one calls
        # estimate() -> _query_db() on this single shared SentenceTransformer. Driving
        # one CUDA module from several threads concurrently killed the process outright
        # (silent exit, no Python traceback - reproduced twice while benchmarking with a
        # 6-thread pool). Embedding is milliseconds; serializing it costs nothing next to
        # the NIM call that follows, and keeps concurrent scans from taking the app down.
        self._embed_lock = threading.Lock()
        self.log(f"RAG Agent ready on {device} (DB contains {self.collection.count()} products)")

    def _query_db(self, query: str, n_results: int = 5):
        with self._embed_lock:
            query_embedding = self.embedding_model.encode([query], show_progress_bar=False).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results

    def _build_context(self, results) -> str:
        context = "Similar Products:\n"
        if not results['metadatas'] or not results['metadatas'][0]:
            return context + "None found.\n"
            
        for i, meta in enumerate(results['metadatas'][0]):
            name = meta.get('name', 'Unknown')
            price = meta.get('price', 0)
            category = meta.get('category', 'Unknown')
            context += f"- {name} (Category: {category}): ₹{price}\n"
        return context

    def estimate(self, item: Item) -> Optional[float]:
        self.log(f"Estimating price for {item.product_title}")
        
        # Build query string
        query = item.describe()
        
        # Get similar products
        results = self._query_db(query)
        context = self._build_context(results)
        
        # Build prompt
        user_prompt = f"Product to estimate:\n{query}\n\n{context}\nWhat is a reasonable price estimate for this product in INR? Respond with ONLY the number."
        
        try:
            response = call_with_model_fallback(
                lambda model: self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0
                ),
                self.models,
                agent_name=self.name,
            )

            result_str = response.choices[0].message.content.strip()
            # Clean up the output just in case
            result_str = result_str.replace("₹", "").replace(",", "").strip()
            estimated_price = float(result_str)
            self.log(f"Estimated price: ₹{estimated_price}")
            return estimated_price

        except Exception as e:
            self.log(f"Error estimating price: {e}")
            return None

    def process(self, deal: Deal) -> Optional[float]:
        try:
            item = self.preprocessor.process(deal)
        except Exception as e:
            self.log(f"Error preprocessing deal: {e}")
            return None
        return self.estimate(item)
