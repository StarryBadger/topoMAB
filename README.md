# TopoMAB System

**Topology-Aware Combinatorial Bandits for Self-Adaptive Code Generation Pipelines**

TopoMAB is a novel Self-Adaptive System (SAS) architecture targeting the software engineering domain. It treats an AI request as a traversal through a Directed Acyclic Graph (DAG) of microservices and utilizes a Contextual Combinatorial Multi-Armed Bandit (CC-MAB) with Semi-Bandit Feedback to dynamically compose the optimal pipeline of prompt strategies, local LLMs, and verification loops.

## Setup

1. Make sure you have `uv` installed (`pip install uv`).
2. Run `uv sync` to install dependencies.
3. Make sure you have Ollama running locally.
4. Pull the required models:
```bash
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:7b
ollama pull codellama:13b
```

## Configuration

Modify the `.env` file to switch between baselines or update the Ollama host:
```env
BASELINE=topomab
OLLAMA_HOST=http://localhost:11434
```
Available baselines: `topomab`, `static_fastest`, `static_most_accurate`, `linucb`, `random`, `epsilon_greedy`.

## Dataset

This project uses the `deepmind/code_contests` dataset. The system streams the dataset natively using the HuggingFace `datasets` library to avoid large upfront downloads. If the dataset cannot be loaded, it will gracefully fallback to a mock dataset for testing purposes.

## Running Experiments

You can use the provided bash script to run all baselines automatically:

```bash
chmod +x scripts/run_experiments.sh
./scripts/run_experiments.sh
```

Alternatively, you can run an individual experiment manually:

```bash
uv run python evaluate.py --policy topomab --train 100 --test 20 --runs 3
```

## File Structure
- `data/`: Dataset loading module.
- `nodes/`: The Microservice DAG components (Prompt, LLM, Verification).
- `bandit/`: The CC-LinUCB and standard LinUCB learning algorithms.
- `mapek/`: Orchestration code for Monitor, Analyze, Plan, Execute, and Knowledge.
- `evaluate.py`: Main evaluation loop containing Phase 1 (Online Learning) and Phase 2 (Hidden Evaluation).
