# TopoMAB: A Topology-Aware Autonomic Architecture for Code Generation

This document details the complete methodology, architectural design, and theoretical foundation of **TopoMAB**, a self-adaptive framework designed to optimize Large Language Model (LLM) code generation pipelines. This text can serve as the direct foundation for the Methodology and Experimental Setup sections of an academic paper.

---

## 1. System Overview
Modern code generation relies on multi-step pipelines (e.g., prompt engineering $\rightarrow$ generation $\rightarrow$ sandbox verification). However, static pipelines fail to account for the heterogeneous complexity of coding problems: simple problems incur unnecessary latency when routed through heavy models, while complex algorithmic problems fail when routed to fast, lightweight models. 

**TopoMAB** solves this by formulating the pipeline routing problem as a **Contextual Combinatorial Multi-Armed Bandit (CC-MAB)** operating under the autonomic computing **MAPE-K** (Monitor, Analyze, Plan, Execute, Knowledge) control loop. The system dynamically adapts the pipeline topology on a per-problem basis to optimize the Pareto frontier of accuracy and execution latency.

---

## 2. Autonomic Architecture (MAPE-K)
The system strictly decouples logic into the autonomic MAPE-K standard, ensuring zero bleeding between telemetry, machine learning, and execution contexts.

### 2.1 Monitor (Sensors & Context Extraction)
The Monitor acts as the environment sensor. When a new problem arrives, it extracts an 8-dimensional contextual feature vector ($x_t \in \mathbb{R}^8$):
* **Quantitative Features**: Problem description length (normalized), computational time limits, and difficulty ratings.
* **Semantic Features (Keywords)**: Binary flags indicating the presence of algorithmic categories (`array`, `graph`, `math`, `tree`, `dynamic`).
The Monitor also measures node-level execution latencies during the Execute phase and handles sandbox test assertions to compute the final accuracy.

### 2.2 Analyze (Bandit Math & Symptom Detection)
The Analyze module acts as the "brain". It receives $x_t$ and queries the Knowledge Base to compute the Upper Confidence Bound (UCB) of expected rewards for all possible pipelines.
* It compares the optimal predicted pipeline against the currently active pipeline.
* If a new pipeline yields a higher expected reward, it generates a "Symptom" declaring that an `adaptation_needed` event has occurred.

### 2.3 Plan (Workflow Generation)
If the Analyze module triggers an `adaptation_needed` event, the Plan module gracefully updates the active workflow configuration in the Knowledge Base. If no adaptation is needed, the Plan module rests, preventing unnecessary configuration overhead.

### 2.4 Execute (Microservice DAG)
The Execute module sequentially routes the problem through the designated nodes of the pipeline Directed Acyclic Graph (DAG), tracking latencies at each boundary.

### 2.5 Knowledge Base (The Blackboard)
The centralized state manager. It stores the active pipeline configuration, historical interaction logs, and the matrices ($A_i$, $b_i$) representing the Ridge Regression weights learned by the Bandit over time.

---

## 3. The Pipeline Action Space
The pipeline is formulated as a DAG with three sequential stages, yielding a combinatorial action space of $2 \times 3 \times 2 = 12$ distinct pipelines.

1. **Prompt Stage**:
   * `zero_shot`: Passes the problem directly to the LLM.
   * `chain_of_thought` (CoT): Augments the prompt requesting step-by-step reasoning in comments.
2. **LLM Stage**:
   * `qwen2.5-coder:1.5b`: Ultra-fast, low-parameter local model.
   * `qwen2.5-coder:7b`: Medium-tier local model.
   * `codellama:13b`: Slower, reasoning-focused local model.
3. **Verification Stage**:
   * `pass_through`: Immediately returns the generated code.
   * `self_reflexion`: Executes the code in a sandbox against public tests. If it fails, the stack trace is appended to a secondary prompt, and the LLM is queried again for a fix.

---

## 4. Learning Algorithm: TopoMAB
TopoMAB is a Contextual Combinatorial Bandit utilizing **Semi-Bandit Feedback**.

### 4.1 Semi-Bandit Formulation
Unlike standard bandits that treat an entire pipeline as a single black-box "arm", TopoMAB exploits the known topology of the DAG. The expected reward of a pipeline $P$ is the sum of the expected rewards of its constituent nodes:
$$ \mathbb{E}[R_P] = \sum_{i \in P} \mathbb{E}[r_i] $$
When a pipeline finishes execution and yields a global reward $R_t$, the reward is evenly apportioned to the nodes that participated. The Ridge Regression matrices for each node $i$ are updated:
$$ A_i \leftarrow A_i + x_t x_t^T $$
$$ b_i \leftarrow b_i + \left( \frac{R_t}{|P|} \right) x_t $$
This drastically accelerates convergence, as learning about the `qwen2.5-coder:7b` node in one pipeline immediately updates beliefs about it in all other pipelines.

### 4.2 Asymmetric Reward Shaping
Code generation on difficult datasets (like CodeContests) presents a highly sparse reward environment (Accuracy is mostly 0.0). To prevent "Reward Hacking" (where the bandit aggressively minimizes latency by picking the fastest model just because all models fail anyway), we implemented an asymmetric, multi-objective formulation:
* **Success ($Acc > 0$)**: $R_t = 10.0 \cdot Acc - w_{lat} \cdot Lat_{norm}$
* **Failure ($Acc == 0$)**: $R_t = -1.0 - w_{lat} \cdot Lat_{norm}$
This forces the Bandit to prioritize finding correct solutions, while still maintaining a gradient that pushes it to "fail fast" if the problem is strictly unsolvable.

### 4.3 Avoiding Test-Case Overfitting
To ensure the Bandit learns true generalization rather than memorizing trivial public examples (a known Proxy Reward Hazard in LLM literature), Phase 1 training feedback is calculated using a subset of the **Hidden Tests** from the `train` split. This zero-leakage approach aligns the training reward gradient precisely with the empirical evaluation metric.

---

## 5. Experimental Evaluation Protocol
Experiments are strictly divided into two temporal phases to emulate CI/CD auto-grader environments:
* **Phase 1 (Online Learning)**: $N=200$ problems streamed from the `train` dataset split. The system actively explores, exploits, and updates its $A$ and $b$ matrices using post-execution feedback.
* **Phase 2 (Hidden Test Evaluation)**: $M=10$ problems streamed from the `valid` dataset split. Learning is frozen. The models must rely entirely on their learned mappings to route problems and are strictly scored on their performance against unseen edge cases.

### 5.1 Baselines
To validate TopoMAB's novelty, it is evaluated against 5 rigorous baselines:
1. **`linucb`**: A standard Contextual Bandit. It treats the 12 pipelines as entirely independent, disjoint arms, failing to share information across shared nodes.
2. **`epsilon_greedy`**: Utilizes TopoMAB's node-level math, but disables UCB exploration in favor of exploiting 90% of the time and uniformly randomizing the pipeline 10% of the time.
3. **`random`**: A stateless baseline that uniformly selects a pipeline at random, serving as the variance control.
4. **`static_fastest`**: A static orchestrator locked to `["zero_shot", "qwen2.5-coder:1.5b", "pass_through"]`. It skips Phase 1 entirely as it possesses no learning mechanisms.
5. **`static_most_accurate`**: A static orchestrator locked to the theoretically optimal pipeline `["chain_of_thought", "codellama:13b", "self_reflexion"]`. It acts as the upper-bound latency control.
