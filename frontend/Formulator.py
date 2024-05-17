import json
import nltk

from typing import Dict, Optional, Union, List, Tuple
from utils.ModelCall import model_call
from frontend import InfoExt, ParamExt, VarExt

formulate_template = ["""
You are an optimization experts who are familiar with optimization problem analyzing and modeling.
""", """
Your are in middle of modeling for an optimization problem.

Here is the background or context of the optimization problem:

{background}

Within the optimization problem, the parameters (input fixed values) are defined as follows:

{parameters}

And the variables (to be determined for optimization) are defined as follows:

{variables}

Now, you are required to generate implicit constraints that the problem implies (e.g., non-negativity of some variables). Then, provide the LaTeX mathematical expression format (do not include the $ symbols) to make the constraint precise and clear.
Take a deep breath and generate a JSON file in the following format:

```json
{{
    "constraints": [
        {{
            "description": "Explanation for the specific constraint",
            "formulation": "The LaTeX mathematical expression (use the parameters and variables mentioned above)"
        }},
        ...
    ]
}}
```
""", """
Your are in middle of modeling for an optimization problem, and currently you are focusing on a part of the problem description. 
Your task is to point out relevant constraints or objective that this part of problem description entails (If Any).
- Notice that possibly neither constraints nor objective could be concluded from this part of description. Also, multiple constraints could be possibly implied.
- Please be careful about the range and indices of involved parameters or variables when generating the formulation.

Here is the background or context of the optimization problem:

{background}

Within the optimization problem, the parameters (input fixed values) are defined as follows:

{parameters}

And the variables (to be determined for optimization) are defined as follows:

{variables}

You are required to extract constraints or objective from given description fragment, and then formulate each constraint or objective that you extracted into LaTeX mathematical expression (do not include the $ symbols) to make it precise and clear.
The description fragment is

{desc_frag}

Now, take a deep breath and generate a JSON file in the following format:

```json
{{
    "constraints": [
        {{
            "description": "Explanation for the specific constraint",
            "formulation": "The LaTeX mathematical expression (use the parameters and variables mentioned above)"
        }},
        ...
    ],
    "objective": [
        {{
            "description": "Explanation for the objective",
            "formulation": "The LaTeX mathematical expression (use the parameters and variables mentioned above)"
        }},
        ...
    ]
}}
```
""", """

In previous rounds, mistakes are made and in consequence, the formulations are not consistent or they can not perfectly reflect all the requirements of the optimization problem.
Here are messages of previous generation, please learn the experience from the messages, follow the instructions and avoid previous mistakes

{history_messages}
"""
]

reflect_template = ["""
You are an optimization experts who are familiar with optimization problem analyzing and modeling.
""", """
You are in middle of modeling for an optimization problem. You task is to evaluate whether the information extracted from the problem descriptions (variables) are consistent or not.

Here is the background or context of the optimization problem:

{background}

Within the optimization problem, the parameters (input fixed values) are defined as follows:

{parameters}

Your responsibility is to evaluate the quality of extracted variables. To be specific, here are some principles for reference
- When the variable should be presented as a tensor, define the integrated tensor with multiple dimensions instead of seperate scalar values (e.g. you should use x with dimension "[n]" instead of x_i as a scalar).
- Items that are already defined as parameters or do not need to be optimized should be erased.

After reviewing, please generate instructions for the extracted information (variables) in the following format

```json
{{
    "variables": {{
        "status": "Consistent / Inconsistent (indicating whether the information is consistent or should be revised)",
        "message": "A brief comment on the extracted information (may contain the reason of the judgement and necessarily instructions to improve)"
    }}
}}
```

- Message should be concrete. For example, you can make a list and explain that which variable is not consistent and how to revise.

The variables (to be determined for optimization) to be evaluated are as follows:

{variables}

Now, take a deep breath and make a thorough review of the modeling of the optimization problem descripted literally.
""", """
You are in middle of modeling for an optimization problem. You task is to evaluate whether the information extracted from the problem descriptions are consistent or not.
To be specific, here are some principles for reference

- Constraints that are redundant (one constraint is equivalent to another) should be pruned.
- For those formulations of constraints or objective that contains variables that are not previously defined, you should consider whether the new variables are necessary or the formulation contains unnecessary components.
- Objective should be unique and precise. If multiple objectives are extracted, they should be merged or pruned.
- The constraints should cover all aspects of the optimization problem, including implicit requirements.
- When constraints or objective involve indices, make sure the formulation is consistent regarding to the meaning of each dimension.

After reviewing the extracted information, please generate instructions for each section of information (variables, constraints, objective) in the following format

```json
{{
    "variables": {{
        "status": "Consistent / Inconsistent (indicating whether the information is consistent or should be revised)",
        "message": "A brief comment on the extracted information (may contain the reason of the judgement and necessarily instructions to improve)"
    }},
    "constraints": {{
        ...
    }},
    "objective": {{
        ...
    }}
}}
```

- Message should be concrete. For example, you can make a list and explain which item is not consistent and how to revise.

Here is the background or context of the optimization problem:

{background}

Within the optimization problem, the parameters (input fixed values) are defined as follows:

{parameters}

And the variables (to be determined for optimization) are defined as follows:

{variables}

The extracted constraints are defined as follows:

{constraints}

And the objective is:

{objective}

Now, take a deep breath and make a thorough review of the modeling of the optimization problem descripted literally.
"""
]

revise_template = ["""
You are an optimization experts who are familiar with optimization problem analyzing and modeling.
""", """
You are given a description of an optimization problem, and your task is to point out relevant {target} that the problem description entails, and formulate them into standard forms (i.e., LaTeX mathematical expression).

Here is the background or context of the optimization problem:

{background}

Previously you have generated a version with mistakes:

{previous}

After review, here is the comments and instructions for improvement:

{message}

- You are expected to follow the previous JSON format when generating the new version:

```json
[
    {{
        "description": "Explanation for the specific constraint",
        "formulation": "The LaTeX mathematical expression (use the parameters and variables mentioned above)"
    }},
    ...
]
```

Now, take a deep breath and generate the JSON file in the required format.
"""]

class Formulator:
    def __init__(
        self, 
        client,
        param_symbols: Optional[List] = None,
        model: Optional[str] = "gpt-3.5-turbo",
        maximum_retries: Optional[int] = 3
    ):
        self.client = client
        self.param_symbols = param_symbols
        self.model = model
        self.maximum_retries = maximum_retries
        
        self.infoExt = InfoExt(client, model)
        self.paramExt = ParamExt(client, model)
        self.varExt = VarExt(client, model)

        self.prob_def = None
        
    def formulate(self, desc: str) -> Dict:
        self.prob_def = dict(
            description=desc,
            background=self.infoExt.extract(desc),
            parameters=self.paramExt.extract(desc, symbols=self.param_symbols),
            variables=[],
            constraints=[],
            objective=[]
        )
        self.prob_def["variables"] = self.varExt.extract(
            desc=desc,
            params=self.prob_def["parameters"]
        )
        print(json.dumps(self.prob_def, indent=4))
        
        for _ in range(self.maximum_retries):
            print(f"Reflect round {_} for var")
            
            decision = self.__reflect("var")
            print("Reflect Results (var)")
            print(json.dumps(decision, indent=4))
            
            if decision["variables"][0] != "Consistent":
                self.prob_def["variables"] = self.varExt.revise(
                    bg=self.prob_def["background"],
                    params=self.prob_def["parameters"],
                    prv=self.prob_def["variables"],
                    msg=decision["variables"][1]
                )
            else:
                break
        
        self.__extract()
        print("Problem Definition")
        print(json.dumps(self.prob_def, indent=4))
        
        for _ in range(self.maximum_retries):
            print(f"Reflect round {_}")
                
            decision = self.__reflect("all")
            print("Reflect Results (all)")
            print(json.dumps(decision, indent=4))
            
            if decision["variables"][0] != "Consistent":
                self.prob_def["variables"] = self.varExt.revise(
                    bg=self.prob_def["background"],
                    params=self.prob_def["parameters"],
                    prv=self.prob_def["variables"],
                    msg=decision["variables"][1]
                )
                self.__extract()
            elif decision["constraints"][0] != "Consistent" or decision["objective"][0] != "Consistent":
                for target in ["constraints", "objective"]:
                    if decision[target][0] != "Consistent":
                        self.prob_def[target] = self.revise(
                            target=target,
                            msg=decision[target][1]
                        )
            else:
                break
            
        return self.prob_def
    
    def __extract(self):
        self.prob_def["constraints"] = []
        self.prob_def["objective"] = []
        
        def __call(prompt_template: str, fragment: Optional[str] = None):
            prompt = prompt_template.format(
                background=self.prob_def["background"],
                parameters=json.dumps(self.prob_def["parameters"], indent=4),
                variables=json.dumps(self.prob_def["variables"], indent=4),
                desc_frag=fragment
            )
            cnt = 3
            while cnt > 0:
                try:
                    response = model_call(
                        client=self.client,
                        prompt=[
                            {"role": "system", "content": formulate_template[0]},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    if "```json" in response:
                        response = response[response.find("```json") + 7 :]
                        response = response[: response.rfind("```")]
                    response = response.replace("```", "").replace("\\", "")
                    response = json.loads(response)
                    break
                
                except Exception as e:
                    print(e)
                    cnt -= 1
                    if cnt == 0:
                        raise RuntimeError("Constriants or objective extraction failed.")
            
            if "constraints" in response:
                self.prob_def["constraints"] += response["constraints"]
            if "objective" in response:
                self.prob_def["objective"] += response["objective"]
            
        __call(formulate_template[1])
        # __call(formulate_template[2], self.prob_def["description"])
        for sent in nltk.sent_tokenize(self.prob_def["description"]):
            __call(formulate_template[2], sent)
        print(json.dumps(self.prob_def, indent=4))
            
    def __reflect(self, target: str) -> List[Tuple]:
        if target == "var":
            prompt = reflect_template[1].format(
                background=self.prob_def["background"],
                parameters=json.dumps(self.prob_def["parameters"], indent=4),
                variables=json.dumps(self.prob_def["variables"], indent=4)
            )
        elif target == "all":
            prompt = reflect_template[2].format(
                background=self.prob_def["background"],
                parameters=json.dumps(self.prob_def["parameters"], indent=4),
                variables=json.dumps(self.prob_def["variables"], indent=4),
                constraints=json.dumps(self.prob_def["constraints"], indent=4),
                objective=json.dumps(self.prob_def["objective"], indent=4)
            )
        cnt = 3
        while cnt > 0:
            try:
                response = model_call(
                    client=self.client,
                    prompt=[
                        {"role": "system", "content": reflect_template[0]},
                        {"role": "user", "content": prompt}
                    ]
                )
                if "```json" in response:
                    response = response[response.find("```json") + 7 :]
                    response = response[: response.rfind("```")]
                response = json.loads(response.replace("```", "").replace("\\", ""))
                return {
                    k: (v["status"], v["message"])
                    for k, v in response.items()
                }
                
            except Exception as e:
                print(e)
                cnt -= 1
                if cnt == 0:
                    raise RuntimeError("Reflection failed.")
        
    def revise(self, target: str, msg: str) -> List:
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
                                background=self.prob_def["background"],
                                target=target,
                                previous=json.dumps(self.prob_def[target], indent=4),
                                message=msg
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
                    raise RuntimeError("Formulator revise failed.")
