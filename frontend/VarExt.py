import json

from typing import Dict, Optional, Union, List
from utils.ModelCall import model_call

prompt_template = ["""
You are an optimization experts who are familiar with optimization problem analyzing and modeling.
""", """
You are given a description of an optimization problem, and your task is to identify all the variables of the problem and provide necessary explanations.
Variables in an optimization problem represent the quantities that must be determined to achieve the desired outcome, which is typically maximizing or minimizing an objective function.
It is important that parameters with given values are not considered to be variables.

You are expected to find out all the variables involved in the description and for each variable, you need to provide:
1. The standard symbol of the variable (e.g., ProductNum, ...)
2. A brief explanation for the variable, determine whether it is continuous, integer or binary
3. Dimension of the variable (e.g., "[]" stands for scalar value and "[N]" or "[N, M]" stands for higher dimensional values)

You should generate a JSON file in the following format:

```json
[
    {{
        "symbol": "The symbol of the variable",
        "description": "Explanation for the variable",
        "dim": "Dimension of the variable (a string)"
    }},
]
```

- When the variable should be presented as a tensor, define the integrated tensor instead of seperate scalar values.

Here is the problem description:

{description}

Here is the parameters extracted from the optimization problem:

{parameters}

Now, take a deep breath and generate the JSON file in the required format.
"""
]

revise_template = ["""
You are an optimization experts who are familiar with optimization problem analyzing and modeling.
""", """
You are given a description of an optimization problem, and your task is to identify all the variables of the problem and provide necessary explanations.
Variables in an optimization problem represent the quantities that must be determined to achieve the desired outcome, which is typically maximizing or minimizing an objective function. Notice that parameters that have fixed values are not considered to be variables.

Here is the background or context of the optimization problem:

{background}

Here is the parameters extracted from the optimization problem:

{parameters}

Previously you have generated a version with mistakes:

{previous}

After review, here is the comments and instructions for improvement:

{message}

- You are expected to follow the previous JSON format when generating the new version:

```json
[
    {{
        "symbol": "The symbol of the variable (e.g., ProductNum, ...)",
        "description": "Explanation for the variable, determine whether it is continuous, integer or binary",
        "dim": "Dimension of the variable (e.g., "[]" stands for scalar value and "[N]" or "[N, M]" stands for higher dimensional values)"
    }},
]
```

Now, take a deep breath and generate the JSON file in the required format.
"""]

class VarExt:
    def __init__(self, client, model: Optional[str] = "gpt-3.5-turbo"):
        self.client=client
        self.model=model
        
    def extract(self, desc: str, params: List) -> List:
        cnt = 3
        while cnt > 0:
            try:
                response = model_call(
                    client=self.client,
                    prompt=[
                        {"role": "system", "content": prompt_template[0]},
                        {
                            "role": "user", 
                            "content": prompt_template[1].format(
                                description=desc,
                                parameters=json.dumps(params, indent=4)
                            )
                        }
                    ],
                    model=self.model
                )
                
                if "```json" in response:
                    response = response[response.find("```json") + 7 :]
                    response = response[: response.rfind("```")]
                response = response.replace("```", "").replace("\\", "")
                return json.loads(response)
                
            except Exception as e:
                print(e)
                cnt -= 1
                if cnt == 0:
                    raise RuntimeError("Variable extraction failed.")
    
    def revise(self, bg: str, params: List, prv: List, msg: str) -> List:
        cnt = 3
        while cnt > 0:
            try:
                response = model_call(
                    client=self.client,
                    prompt=[
                        {"role": "system", "content": revise_template[0]},
                        {
                            "role": "user", 
                            "content": revise_template[1].format(
                                background=bg,
                                parameters=json.dumps(params, indent=4),
                                previous=json.dumps(prv, indent=4),
                                message=msg)
                        }
                    ],
                    model=self.model
                )
                
                if "```json" in response:
                    response = response[response.find("```json") + 7 :]
                    response = response[: response.rfind("```")]
                response = response.replace("```", "").replace("\\", "")
                return json.loads(response)
            
            except Exception as e:
                print(e)
                cnt -= 1
                if cnt == 0:
                    raise RuntimeError("Variable revise failed.")