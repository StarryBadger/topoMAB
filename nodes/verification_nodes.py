def pass_through(code: str, problem: dict, **kwargs) -> str:
    """Outputs the code immediately (0 latency overhead)."""
    return code

def self_reflexion(code: str, problem: dict, monitor=None, llm_generate_func=None, **kwargs) -> str:
    """Runs code against visible tests. If it fails, asks the LLM to generate a fix."""
    tests = problem.get('public_tests', {})
    if not monitor or not llm_generate_func:
        return code
        
    acc = monitor.evaluate_code(code, tests)
    if acc < 1.0:
        fix_prompt = f"The following code failed some tests for the problem. Please fix it.\n\nCode:\n{code}\n\nProblem:\n{problem.get('description', '')}\n\nOutput only the fixed Python code."
        fixed_code = llm_generate_func(fix_prompt)
        return fixed_code
    return code

# Mapping to easily fetch the function by name
VERIFICATION_REGISTRY = {
    "pass_through": pass_through,
    "self_reflexion": self_reflexion
}
