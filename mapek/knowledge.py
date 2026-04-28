import json
import os
import numpy as np
from typing import Dict, Any, List

class KnowledgeBase:
    """The shared blackboard for the MAPE-K loop."""
    
    def __init__(self, db_path="data/knowledge.json", d=3, nodes=None):
        self.db_path = db_path
        self.history = []
        
        if nodes is None:
            nodes = []
            
        # Bandit state mapping nodes to their Ridge Regression parameters
        self.A = {node: np.identity(d) for node in nodes}
        self.b = {node: np.zeros((d, 1)) for node in nodes}
        
        # Current active pipeline (Workflow)
        self.current_pipeline = ["zero_shot", "qwen2.5-coder:1.5b", "pass_through"]
        
        # Global hyperparameters
        self.alpha = 1.0
        self.w_lat = 0.1
        self.w_acc = 1.0
        self.max_lat = 1.0

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    self.history = json.load(f)
            except:
                self.history = []

    def log_interaction(self, problem_id: str, context: List[float], pipeline: List[str], latencies: Dict[str, float], accuracy: float, reward: float):
        entry = {
            "problem_id": problem_id,
            "context": context,
            "pipeline": pipeline,
            "latencies": latencies,
            "accuracy": accuracy,
            "reward": reward
        }
        self.history.append(entry)
        
    def save(self):
        with open(self.db_path, "w") as f:
            json.dump(self.history, f, indent=2)
