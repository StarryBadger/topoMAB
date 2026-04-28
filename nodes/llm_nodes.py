import os
import time
import ollama

def generate_code_qwen_1_5b(prompt: str) -> str:
    return _generate_code(prompt, "qwen2.5-coder:1.5b")

def generate_code_qwen_7b(prompt: str) -> str:
    return _generate_code(prompt, "qwen2.5-coder:7b")

def generate_code_codellama_13b(prompt: str) -> str:
    return _generate_code(prompt, "codellama:13b")

def _generate_code(prompt: str, model_name: str) -> str:
    """Generates code using Ollama API."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    client = ollama.Client(host=host)
    
    try:
        response = client.generate(model=model_name, prompt=prompt, options={"temperature": 0.2})
        code = response['response']
    except Exception as e:
        code = f"# Error generating code: {e}"
        
    # Extract code from markdown block if present
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
        
    return code

# Mapping to easily fetch the function by name
LLM_REGISTRY = {
    "qwen2.5-coder:1.5b": generate_code_qwen_1_5b,
    "qwen2.5-coder:7b": generate_code_qwen_7b,
    "codellama:13b": generate_code_codellama_13b
}
