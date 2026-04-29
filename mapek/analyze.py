import numpy as np
from typing import List, Dict, Any
from mapek.knowledge import KnowledgeBase
import random

class Analyze:
    """Analyzes context, computes rewards, and detects if an adaptation is needed."""
    
    def __init__(self, pipelines: List[List[str]]):
        self.pipelines = pipelines

    def _get_node_ucb(self, node: str, x_vec: np.ndarray, kb: KnowledgeBase) -> float:
        A_inv = np.linalg.inv(kb.A[node])
        theta = A_inv @ kb.b[node]
        
        # Dynamic Decaying Exploration
        t = max(1, getattr(kb, 'timestep', 1))
        dynamic_alpha = kb.alpha / np.sqrt(t)
        
        p = theta.T @ x_vec + dynamic_alpha * np.sqrt(x_vec.T @ A_inv @ x_vec)
        return float(p[0, 0])

    def detect_symptoms(self, context: List[float], kb: KnowledgeBase, policy: str) -> Dict[str, Any]:
        """
        Determines if adaptation is needed based on the active policy logic.
        """
        x_vec = np.array(context).reshape((-1, 1))
        
        if policy == "epsilon_greedy":
            if random.random() < 0.1:
                # 10% chance to pick a completely random pipeline
                best_pipeline = random.choice(self.pipelines)
                if best_pipeline != kb.current_pipeline:
                    return {"adaptation_needed": True, "new_pipeline": best_pipeline}
                return {"adaptation_needed": False}
            else:
                # 90% chance to exploit using TopoMAB logic
                policy = "topomab"
                
        max_ucb = -float('inf')
        best_pipeline = None
        
        # Check expected reward for all pipelines
        for pipeline in self.pipelines:
            if policy == "topomab":
                # Semi-bandit: Sum of individual node UCBs
                pipeline_ucb = sum(self._get_node_ucb(node, x_vec, kb) for node in pipeline)
            elif policy == "linucb":
                # Standard bandit: The pipeline itself is treated as a single arm
                pipeline_name = "_".join(pipeline)
                pipeline_ucb = self._get_node_ucb(pipeline_name, x_vec, kb)
                
            if pipeline_ucb > max_ucb:
                max_ucb = pipeline_ucb
                best_pipeline = pipeline
                
        # If the reward is highest for current model in use, decide not to do anything.
        if best_pipeline != kb.current_pipeline:
            return {"adaptation_needed": True, "new_pipeline": best_pipeline}
        
        # Symptom: No adaptation needed
        return {"adaptation_needed": False}

    def calculate_reward(self, accuracy: float, node_latencies: Dict[str, float], kb: KnowledgeBase) -> float:
        """Calculates scalar reward using asymmetric multi-objective formulation."""
        total_latency = sum(node_latencies.values())
        if total_latency > kb.max_lat:
            kb.max_lat = total_latency
        
        norm_lat = total_latency / kb.max_lat if kb.max_lat > 0 else 0
        
        # Asymmetric Reward Shaping
        if accuracy == 0.0:
            reward = -1.0 # Flat penalty for failing
        else:
            reward = 10.0 * accuracy - kb.w_lat * norm_lat
            
        return reward

    def update_knowledge(self, context: List[float], pipeline: List[str], reward: float, kb: KnowledgeBase, policy: str):
        """Updates internal models in the Knowledge Base."""
        x_vec = np.array(context).reshape((-1, 1))
        
        if policy in ["topomab", "epsilon_greedy"]:
            # Semi-bandit feedback: Apportion reward to the specific nodes used
            for node in pipeline:
                node_reward = reward / len(pipeline)
                kb.A[node] += x_vec @ x_vec.T
                kb.b[node] += node_reward * x_vec
        elif policy == "linucb":
            # Standard bandit feedback: Update the entire pipeline arm
            pipeline_name = "_".join(pipeline)
            kb.A[pipeline_name] += x_vec @ x_vec.T
            kb.b[pipeline_name] += reward * x_vec
            
        if not hasattr(kb, 'timestep'):
            kb.timestep = 0
        kb.timestep += 1
