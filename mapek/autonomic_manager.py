from typing import Dict, Any, List

from mapek.monitor import Monitor
from mapek.analyze import Analyze
from mapek.plan import Plan
from mapek.execute import Execute
from mapek.knowledge import KnowledgeBase

class AutonomicManager:
    """Event-driven orchestrator for the MAPE-K loop."""
    
    def __init__(self, knowledge: KnowledgeBase, pipelines: List[List[str]], policy: str = "topomab"):
        self.knowledge = knowledge
        self.monitor = Monitor()
        self.analyze = Analyze(pipelines)
        self.plan = Plan()
        self.execute = Execute(self.monitor)
        self.policy = policy
        self.pipelines = pipelines

    def handle_request(self, problem: Dict[str, Any]) -> str:
        # 1. Monitor parses context from the incoming problem
        context = self.monitor.extract_context(problem)
        
        # 2. Analyze checks if adaptation is needed (is current pipeline still best?)
        if self.policy in ["topomab", "linucb", "epsilon_greedy"]:
            symptoms = self.analyze.detect_symptoms(context, self.knowledge, self.policy)
        else:
            symptoms = self._static_policy_symptoms(context)
            
        # 3. Plan builds or updates the workflow if symptoms indicate adaptation needed
        workflow = self.plan.create_workflow(symptoms, self.knowledge)
        
        # 4. Execute runs the code via the selected workflow
        code = self.execute.run(workflow, problem)
        
        return code, context, workflow
        
    def process_feedback(self, problem: Dict[str, Any], code: str, context: List[float], workflow: List[str]):
        """Processes the feedback after the visible tests are run to update the Knowledge base."""
        # Evaluate execution correctness on visible tests
        visible_tests = problem.get('public_tests', {})
        acc = self.monitor.evaluate_code(code, visible_tests)
        
        latencies = self.monitor.node_latencies.copy()
        
        # 5/6. Analyze calculates reward and updates learning models in Knowledge Base
        reward = self.analyze.calculate_reward(acc, latencies, self.knowledge)
        
        if self.policy in ["topomab", "linucb", "epsilon_greedy"]:
            self.analyze.update_knowledge(context, workflow, reward, self.knowledge, self.policy)
            
        self.knowledge.log_interaction(
            problem.get("name", "N/A"), context, workflow, 
            latencies, acc, reward
        )
        return acc

    def _static_policy_symptoms(self, context):
        """Helper to bypass bandit for standard baselines."""
        import random
        if self.policy == "static_fastest":
            new_pipe = ["zero_shot", "qwen2.5-coder:1.5b", "pass_through"]
        elif self.policy == "static_most_accurate":
            new_pipe = ["chain_of_thought", "codellama:13b", "self_reflexion"]
        elif self.policy == "random":
            new_pipe = random.choice(self.pipelines)
        else:
            new_pipe = self.knowledge.current_pipeline
            
        if new_pipe != self.knowledge.current_pipeline:
            return {"adaptation_needed": True, "new_pipeline": new_pipe}
        return {"adaptation_needed": False}
