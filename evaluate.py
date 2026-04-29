import os
import argparse
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from data.dataset_handler import CodeContestsHandler
from mapek.knowledge import KnowledgeBase
from mapek.autonomic_manager import AutonomicManager
from nodes.prompt_nodes import PROMPT_REGISTRY
from nodes.llm_nodes import LLM_REGISTRY
from nodes.verification_nodes import VERIFICATION_REGISTRY

load_dotenv()

def get_all_pipelines():
    prompt_nodes = list(PROMPT_REGISTRY.keys())
    llm_nodes = list(LLM_REGISTRY.keys())
    verif_nodes = list(VERIFICATION_REGISTRY.keys())
    pipelines = []
    for p in prompt_nodes:
        for l in llm_nodes:
            for v in verif_nodes:
                pipelines.append([p, l, v])
    return pipelines, prompt_nodes + llm_nodes + verif_nodes

def run_experiment(policy: str, num_train: int, num_test: int, run_idx: int):
    """Runs a single experiment (train + test)."""
    print(f"--- Running {policy} (Run {run_idx}) ---")
    
    pipelines, all_nodes = get_all_pipelines()
    
    # Generate pipeline-level arms for linucb
    pipeline_names = ["_".join(p) for p in pipelines]
    all_keys = all_nodes + pipeline_names
    
    # Initialize components
    knowledge = KnowledgeBase(db_path=f"data/knowledge_{policy}_{run_idx}.json", d=3, nodes=all_keys)
    manager = AutonomicManager(knowledge, pipelines, policy=policy)
    
    train_handler = CodeContestsHandler(split="train")
    test_handler = CodeContestsHandler(split="valid")
    
    # Determine if this policy actually learns anything
    DO_NOT_LEARN_POLICIES = ["static_fastest", "static_most_accurate", "random"]
    skip_learning = policy in DO_NOT_LEARN_POLICIES
    
    # Phase 1: Online Learning Simulation
    if not skip_learning:
        print("Phase 1: Online Learning")
        for i in tqdm(range(num_train)):
            problem = train_handler.get_next_problem()
            if not problem:
                break
                
            # MAPE-K Event Loop Handles Request
            code, context, workflow = manager.handle_request(problem)
            
            # Reward generation and Knowledge update based on visible tests
            manager.process_feedback(problem, code, context, workflow)
    else:
        print(f"Phase 1: Online Learning (Skipped - {policy} does not learn)")
        
    # Phase 2: Final Scoring (Empirical Study)
    print("Phase 2: Evaluation on Hidden Tests")
    eval_latencies = []
    eval_accuracies = []
    
    for i in tqdm(range(num_test)):
        problem = test_handler.get_next_problem()
        if not problem:
            break
            
        # MAPE-K Event Loop Handles Request (No learning feedback here)
        code, context, workflow = manager.handle_request(problem)
        
        # Evaluate on hidden test cases!
        hidden_tests = problem.get('private_tests', {})
        acc = manager.monitor.evaluate_code(code, hidden_tests)
        
        total_latency = sum(manager.monitor.node_latencies.values())
        eval_latencies.append(total_latency)
        eval_accuracies.append(acc)
        
    knowledge.save()
    
    avg_latency = np.mean(eval_latencies) if eval_latencies else 0
    avg_accuracy = np.mean(eval_accuracies) if eval_accuracies else 0
    
    print(f"Results for {policy} (Run {run_idx}):")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"Avg Pass Rate (Hidden Tests): {avg_accuracy:.2f}")
    
    return avg_latency, avg_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TopoMAB Experiments")
    parser.add_argument("--policy", type=str, default="topomab", help="Policy to run")
    parser.add_argument("--train", type=int, default=50, help="Number of train problems")
    parser.add_argument("--test", type=int, default=10, help="Number of test problems")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to run for averaging")
    
    args = parser.parse_args()
    
    policy_override = os.environ.get("BASELINE")
    if policy_override:
        args.policy = policy_override
        
    all_lats = []
    all_accs = []
    
    for r in range(args.runs):
        lat, acc = run_experiment(args.policy, args.train, args.test, r+1)
        all_lats.append(lat)
        all_accs.append(acc)
        
    print(f"\n=== Final Averaged Results for {args.policy} over {args.runs} runs ===")
    print(f"Latency: {np.mean(all_lats):.2f} ± {np.std(all_lats):.2f}s")
    print(f"Accuracy: {np.mean(all_accs):.2f} ± {np.std(all_accs):.2f}")
