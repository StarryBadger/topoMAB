def zero_shot(problem: dict) -> str:
    """Returns the zero-shot prompt."""
    desc = problem.get("description", "")
    return f"Solve the following programming problem in Python. Output only the code.\n\n{desc}"

def chain_of_thought(problem: dict) -> str:
    """Returns the Chain-of-Thought prompt."""
    desc = problem.get("description", "")
    return f"Solve the following programming problem in Python. Let's think step by step. First, explain the logic in comments, then write the code. Output only the code block.\n\n{desc}"

# Mapping to easily fetch the function by name
PROMPT_REGISTRY = {
    "zero_shot": zero_shot,
    "chain_of_thought": chain_of_thought
}
