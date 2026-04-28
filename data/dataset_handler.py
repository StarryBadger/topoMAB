import os
import json
from datasets import load_dataset
from typing import Dict, Any

class CodeContestsHandler:
    """Handles streaming CodeContests dataset for training and evaluation."""
    
    def __init__(self, split="train", streaming=True):
        self.split = split
        try:
            self.dataset = load_dataset("deepmind/code_contests", split=split, streaming=streaming)
            self.iterator = iter(self.dataset)
            self.use_mock = False
        except Exception as e:
            print(f"Warning: Failed to load huggingface dataset ({e}). Using mock data.")
            self.use_mock = True
            self.mock_index = 0

    def get_next_problem(self) -> Dict[str, Any]:
        """Returns the next problem from the dataset."""
        if self.use_mock:
            # Generate dummy data for testing the pipeline if dataset fails to load
            self.mock_index += 1
            if self.mock_index > 20: # Mock dataset size
                return None
            return {
                "name": f"Mock Problem {self.mock_index}",
                "description": f"Write a program that takes an integer and outputs its square. Problem ID: {self.mock_index}",
                "public_tests": {
                    "input": ["2\n", "3\n"],
                    "output": ["4\n", "9\n"]
                },
                "private_tests": {
                    "input": ["4\n", "5\n"],
                    "output": ["16\n", "25\n"]
                },
                "time_limit": {"seconds": 2.0},
                "difficulty": 1
            }

        try:
            row = next(self.iterator)
            return row
        except StopIteration:
            return None
