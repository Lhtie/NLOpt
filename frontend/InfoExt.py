from typing import Dict, Optional, Union, List
from utils.ModelCall import model_call

prompt_template = ["""
You are an optimization experts who are familiar with optimization problem analyzing and modeling.
""", """
You are given a description of an optimization problem, and your task is to understand the problem description and provide a concise summary of the problem.

The summary you generated should contain the following information:
1. The basic background and context of the problem.
2. Important details that may change the definition of the problem. (Can be omitted)

You are supposed to generate the summary directly in several sentences without any auxiliary words or explanations.

Here is the problem description:

{description}

Please take a deep breath and generate the summary.
"""
]

class InfoExt:
    def __init__(self, client, model: Optional[str] = "gpt-3.5-turbo"):
        self.client = client
        self.model = model
        
    def extract(self, desc: str) -> str:
        response = model_call(
            client=self.client,
            prompt=[
                {"role": "system", "content": prompt_template[0]},
                {"role": "user", "content": prompt_template[1].format(description=desc)}
            ],
            model=self.model
        )
        
        return response