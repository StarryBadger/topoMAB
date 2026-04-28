import numpy as np
from typing import List, Dict

class CCLinUCB:
    """Contextual Combinatorial LinUCB with Semi-Bandit Feedback."""
    
    def __init__(self, d: int, alpha: float, node_names: List[str]):
        self.d = d
        self.alpha = alpha
        self.nodes = node_names
        
        # Ridge regression parameters per node
        self.A = {node: np.identity(d) for node in self.nodes}
        self.b = {node: np.zeros((d, 1)) for node in self.nodes}

    def get_ucb(self, node: str, x_vec: np.ndarray) -> float:
        """Calculates Upper Confidence Bound for a specific node."""
        A_inv = np.linalg.inv(self.A[node])
        theta = A_inv @ self.b[node]
        p = theta.T @ x_vec + self.alpha * np.sqrt(x_vec.T @ A_inv @ x_vec)
        return float(p[0, 0])

    def select_pipeline(self, context: List[float], pipelines: List[List[str]]) -> List[str]:
        """Selects the pipeline that maximizes the sum of node UCBs."""
        x_vec = np.array(context).reshape((self.d, 1))
        
        best_pipeline = None
        max_ucb = -float('inf')
        
        for pipeline in pipelines:
            pipeline_ucb = sum(self.get_ucb(node, x_vec) for node in pipeline)
            if pipeline_ucb > max_ucb:
                max_ucb = pipeline_ucb
                best_pipeline = pipeline
                
        return best_pipeline

    def update(self, context: List[float], pipeline: List[str], reward: float, node_latencies: Dict[str, float], w_lat: float):
        """Updates internal models based on Semi-Bandit feedback."""
        x_vec = np.array(context).reshape((self.d, 1))
        
        for node in pipeline:
            # Simple attribution: base reward share
            node_reward = reward / len(pipeline)
            self.A[node] += x_vec @ x_vec.T
            self.b[node] += node_reward * x_vec

class StandardLinUCB:
    """Standard Contextual Bandit (treating each pipeline as a separate black-box arm)."""
    
    def __init__(self, d: int, alpha: float, pipelines: List[List[str]]):
        self.d = d
        self.alpha = alpha
        self.arms = ["_".join(p) for p in pipelines]
        
        self.A = {arm: np.identity(d) for arm in self.arms}
        self.b = {arm: np.zeros((d, 1)) for arm in self.arms}

    def select_pipeline(self, context: List[float], pipelines: List[List[str]]) -> List[str]:
        x_vec = np.array(context).reshape((self.d, 1))
        
        best_pipeline = None
        max_ucb = -float('inf')
        
        for pipeline in pipelines:
            arm = "_".join(pipeline)
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            p = theta.T @ x_vec + self.alpha * np.sqrt(x_vec.T @ A_inv @ x_vec)
            ucb = float(p[0, 0])
            
            if ucb > max_ucb:
                max_ucb = ucb
                best_pipeline = pipeline
                
        return best_pipeline

    def update(self, context: List[float], pipeline: List[str], reward: float, *args):
        x_vec = np.array(context).reshape((self.d, 1))
        arm = "_".join(pipeline)
        
        self.A[arm] += x_vec @ x_vec.T
        self.b[arm] += reward * x_vec
