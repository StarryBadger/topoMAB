import numpy as np
from typing import List, Dict, Any
from mapek.knowledge import KnowledgeBase

class Analyze:
    """Analyzes context, computes rewards, and detects if an adaptation is needed."""
    
    def __init__(self, pipelines: List[List[str]]):
        self.pipelines = pipelines

    def _get_node_ucb(self, node: str, x_vec: np.ndarray, kb: KnowledgeBase) -> float:
        A_inv = np.linalg.inv(kb.A[node])
        theta = A_inv @ kb.b[node]
        p = theta.T @ x_vec + kb.alpha * np.sqrt(x_vec.T @ A_inv @ x_vec)
        return float(p[0, 0])

    def detect_symptoms(self, context: List[float], kb: KnowledgeBase) -> Dict[str, Any]:
        """
        Determines if adaptation is needed by comparing the expected reward (UCB) 
        of the current pipeline against all other pipelines.
        """
        x_vec = np.array(context).reshape((-1, 1))
        
        # Calculate expected reward (UCB) for current active pipeline
        current_ucb = sum(self._get_node_ucb(node, x_vec, kb) for node in kb.current_pipeline)
        
        best_pipeline = kb.current_pipeline
        max_ucb = current_ucb
        
        # Check if any other pipeline has a HIGHER expected reward
        for pipeline in self.pipelines:
            pipeline_ucb = sum(self._get_node_ucb(node, x_vec, kb) for node in pipeline)
            if pipeline_ucb > max_ucb:
                max_ucb = pipeline_ucb
                best_pipeline = pipeline
                
        # If the reward is highest for current model in use, decide not to do anything.
        if best_pipeline != kb.current_pipeline:
            return {"adaptation_needed": True, "new_pipeline": best_pipeline}
        
        # Symptom: No adaptation needed
        return {"adaptation_needed": False}

    def calculate_reward(self, accuracy: float, node_latencies: Dict[str, float], kb: KnowledgeBase) -> float:
        """Calculates scalar reward using multi-objective formulation."""
        total_latency = sum(node_latencies.values())
        if total_latency > kb.max_lat:
            kb.max_lat = total_latency
        
        norm_lat = total_latency / kb.max_lat if kb.max_lat > 0 else 0
        reward = kb.w_acc * accuracy - kb.w_lat * norm_lat
        return reward

    def update_knowledge(self, context: List[float], pipeline: List[str], reward: float, kb: KnowledgeBase):
        """Updates internal models in the Knowledge Base based on Semi-Bandit feedback."""
        x_vec = np.array(context).reshape((-1, 1))
        
        for node in pipeline:
            node_reward = reward / len(pipeline)
            kb.A[node] += x_vec @ x_vec.T
            kb.b[node] += node_reward * x_vec
