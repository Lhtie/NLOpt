import json

from typing import Dict, Optional, Union, List
from utils.ModelCall import model_call

prompt_template = ["""
You are an optimization experts who are familiar with optimization problem analyzing and modeling.
""", """
You are given a description of an optimization problem and the parameters involved, and your task is to provide necessary explanations for these parameters.
Parameters in an optimization problem refer to the specific values or characteristics that influence the problem but are not determined by the optimization process itself. These values are fixed initially and help define the constraints and objectives of the optimization model.

You are given a list of symbols that refer to parameters in the problem, and your responsibility is to provide descriptions for them in the following format:
1. The standard symbol of the parameter
2. A brief explanation for the parameter
3. Dimension of the parameter

You should generate a JSON file in the following format:

```json
[
    {{
        "symbol": "The symbol of the parameter",
        "description": "Explanation for the parameter",
        "dim": "Dimension of the parameter (string)"
    }},
]
```

Here is the problem description:

{description}

Here is a list of parameters with symbols and their dimensions in the format [(symbol, dim), ...]:

{symbols}

Now, take a deep breath and generate the JSON file in the required format.
"""]

class ParamExt:
    def __init__(self, client, model: Optional[str] = "gpt-3.5-turbo"):
        self.client=client
        self.model=model
        
    def extract(self, desc: str, symbols: Optional[List] = None) -> List:
        if symbols is None:
            prompt = [
                {"role": "system", "content": prompt_template[0]},
                {"role": "user", "content": prompt_template[1].format(description=desc)}
            ]
        else:
            prompt = [
                {"role": "system", "content": prompt_template[0]},
                {
                    "role": "user", 
                    "content": prompt_template[1].format(description=desc, symbols=symbols)
                }
            ]
        
        cnt = 5
        while cnt > 0:
            try:
                response = model_call(
                    client=self.client,
                    prompt=prompt,
                    model=self.model
                )
                if "```json" in response:
                    response = response[response.find("```json") + 7 :]
                    response = response[: response.rfind("```")]
                response = response.replace("```", "").replace("\\", "")
                response = json.loads(response)
                
                for (param, ctx) in zip(response, symbols):
                    assert param['symbol'] == ctx[0]
                    assert param['dim'] == ctx[1]
                return response
                
            except Exception as e:
                print(e)
                cnt -= 1
                if cnt == 0:
                    raise RuntimeError("Parameter extraction failed.")
        