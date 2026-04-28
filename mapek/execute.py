from typing import List, Dict, Any
from nodes.prompt_nodes import PROMPT_REGISTRY
from nodes.llm_nodes import LLM_REGISTRY
from nodes.verification_nodes import VERIFICATION_REGISTRY
from mapek.monitor import Monitor

class Execute:
    """Carries out the planned workflow (the pipeline execution)."""
    
    def __init__(self, monitor: Monitor):
        self.monitor = monitor

    def run(self, pipeline: List[str], problem: Dict[str, Any]) -> str:
        """Executes the pipeline nodes sequentially and times them via the Monitor."""
        self.monitor.clear_latencies()
        p_node, l_node, v_node = pipeline
        
        # Stage 1: Prompt
        t0 = self.monitor.start_timer(p_node)
        prompt = PROMPT_REGISTRY[p_node](problem)
        self.monitor.stop_timer(p_node, t0)
        
        # Stage 2: LLM
        t0 = self.monitor.start_timer(l_node)
        code = LLM_REGISTRY[l_node](prompt)
        self.monitor.stop_timer(l_node, t0)
        
        # Stage 3: Verification
        t0 = self.monitor.start_timer(v_node)
        def llm_generate_func(fix_prompt):
            return LLM_REGISTRY[l_node](fix_prompt)
            
        final_code = VERIFICATION_REGISTRY[v_node](
            code=code, 
            problem=problem, 
            monitor=self.monitor, 
            llm_generate_func=llm_generate_func
        )
        self.monitor.stop_timer(v_node, t0)
        
        return final_code
