import time
import subprocess
import tempfile
from typing import Dict, Any, List

class Monitor:
    """Senses the environment and measures telemetry."""
    
    def __init__(self):
        self.node_latencies = {}

    def extract_context(self, problem: Dict[str, Any]) -> List[float]:
        """Extracts context features Xt from problem."""
        desc = problem.get('description', '')
        desc_len = len(desc)
        
        time_limit_dict = problem.get('time_limit', {})
        time_limit = float(time_limit_dict.get('seconds', 2.0)) if isinstance(time_limit_dict, dict) else 2.0
        
        difficulty = float(problem.get('difficulty', 1))
        
        return [desc_len / 5000.0, time_limit / 5.0, difficulty / 5.0]

    def start_timer(self, node_name: str) -> float:
        return time.time()

    def stop_timer(self, node_name: str, start_time: float):
        latency = time.time() - start_time
        self.node_latencies[node_name] = latency
        return latency

    def evaluate_code(self, code: str, tests: Dict[str, Any]) -> float:
        """Evaluates the code against tests in a sandbox to get Accuracy score."""
        if not code or "# Error" in code:
            return 0.0
            
        inputs = tests.get('input', [])
        outputs = tests.get('output', [])
        
        total = len(inputs)
        if total == 0:
            return 0.0
            
        passed = 0
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as temp:
            temp.write(code)
            temp.flush()
            
            for i in range(total):
                inp = inputs[i]
                expected = outputs[i]
                
                try:
                    process = subprocess.run(
                        ['python', temp.name],
                        input=inp,
                        text=True,
                        capture_output=True,
                        timeout=2.0
                    )
                    if process.returncode == 0 and process.stdout.strip() == expected.strip():
                        passed += 1
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass
                    
        return passed / total
        
    def clear_latencies(self):
        self.node_latencies = {}
