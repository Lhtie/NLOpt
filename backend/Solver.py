import json

from typing import Dict, Optional, Union, List, Tuple
from backend import Programmer, Evaluator

class Solver:
    def __init__(
        self, 
        client, 
        data_json_path: str, 
        model: Optional[str] = "gpt-3.5-turbo",
        maximum_retries: Optional[int] = 3
    ):
        self.client = client
        self.data_json_path = data_json_path
        self.model = model
        self.maximum_retries = maximum_retries
        
    def solve(self, state: Dict) -> (str, Dict):
        state["data_json_path"] = self.data_json_path
        state["sol_status"] = None
        programmer = Programmer(client=self.client, model=self.model)
        evaluator = Evaluator(client=self.client, model=self.model)
        
        for _ in range(self.maximum_retries):
            res, state = programmer.program(state)
            print(res)
            
            res, state = evaluator.eval(state)
            print(res)
            
            if state["sol_status"] == "solved":
                break
        
        if state["sol_status"] == "solved":
            return "success", state
        else:
            return "fail", state