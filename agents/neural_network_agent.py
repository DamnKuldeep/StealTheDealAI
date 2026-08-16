import os
import threading
import torch
from typing import Optional

from agents.deals import Deal
from agents.items import Item
from agents.preprocessor import Preprocessor
from agents.base_agent import Agent
from agents.deep_neural_network import DeepNeuralNetwork, get_vectorizer, inverse_transform_price
from config import settings


class NeuralNetworkAgent(Agent):
    """
    Neural Network Agent: Loads a local PyTorch model to estimate prices.
    """
    name = "Neural Network Agent"
    color = Agent.GREEN

    def __init__(self):
        self.log(f"Initializing Neural Network Agent from {settings.DNN_WEIGHTS_PATH}")
        self.preprocessor = Preprocessor()
        
        self.vectorizer = get_vectorizer()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # EnsembleAgent is a single shared instance reused across every deal, and
        # PlanningAgent evaluates MAX_CONCURRENT_DEALS deals at once - so up to that many
        # threads can call estimate() -> self.model(...) on this one model concurrently.
        # FrontierAgent's SentenceTransformer hit exactly this pattern and crashed the
        # whole process outright (silent exit, no traceback) when driven from several
        # threads at once; serializing the forward pass costs nothing next to the NIM/
        # Modal calls running alongside it, and closes off the same risk here.
        self._infer_lock = threading.Lock()

        try:
            self.model = DeepNeuralNetwork(input_size=5000, hidden_size=2048)
            if os.path.exists(settings.DNN_WEIGHTS_PATH):
                self.model.load_state_dict(torch.load(settings.DNN_WEIGHTS_PATH, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.log(f"Successfully loaded DNN weights onto {self.device}")
            else:
                self.model = None
                self.log(f"Warning: DNN weights not found at {settings.DNN_WEIGHTS_PATH}")
        except Exception as e:
            self.log(f"Error loading DNN model: {e}")
            self.model = None

    def estimate(self, item: Item) -> Optional[float]:
        if not self.model:
            self.log("Cannot estimate: DNN model not loaded")
            return None

        self.log(f"Estimating price for {item.product_title}")
        
        # Build description string as done in training
        desc = f"{item.product_title} {item.product_category}  {item.product_description}"
        
        try:
            # Vectorize
            X = self.vectorizer.transform([desc]).toarray()
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Predict
            with self._infer_lock, torch.no_grad():
                y_pred_scaled = self.model(X_tensor).item()

            # Inverse transform
            estimated_price = inverse_transform_price(y_pred_scaled)
            self.log(f"Estimated price: ₹{estimated_price:.2f}")
            return estimated_price
            
        except Exception as e:
            self.log(f"Error estimating price via DNN: {e}")
            return None

    def process(self, deal: Deal) -> Optional[float]:
        try:
            item = self.preprocessor.process(deal)
        except Exception as e:
            self.log(f"Error preprocessing deal: {e}")
            return None
        return self.estimate(item)
