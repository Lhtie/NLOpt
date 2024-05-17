import json

from typing import Dict, Optional
from utils.ModelCall import model_call

variable_definition_prompt_templates = [
"""
You're an expert programmer who are familiar with optimization problem modeling. Your responsibility is to write Python code for defining variables of the problem.
""", """
Assume the parameters are defined, and generate a code accordingly. Only generate the code, and don't generate any other text. Here's an example:

**input**:

{{
    "symbol": "buy",
    "description": "Quantity of oil i bought in month m",
    "dim": ["I","M"]
}}

***output***:

```
buy = model.addVars(I, M, vtype=gp.GRB.CONTINUOUS, name="buy")
```

- Use model.addVar instead of model.addVars if the variable is a scalar.

Here's a variable that you are expected to write the code:

-----
{variable}
-----

Take a deep breath and generate the code accordingly.
""",
]

main_prompt_templates = {
    "constraints": [
"""
You're an expert programmer who are familiar with optimization problem modeling. Your responsibility is to write Python code for defining constraints of the problem.
""", """
- Assume the parameters and variables are defined, and gurobipy is imported as gp. Generate a code accordingly, and don't generate any other text.
- If the constraint requires changing a variable's integralilty, generate the code for changing the variable's integrality rather than defining the variable again.
- Variables should become before parameters when defining inequality constraints in gurobipy (because of the gurobi parsing order syntax).
- If there is no code needed, just generate the comment line (using # ) explaining why.

Here's an example:


**input**:


{{
    "description": "in month m, it is possible to store up to storageSize_{{m}} tons of each raw oil for use later.",
    "formulation": "\(storage_{{i,m}} \leq storageSize, \quad \\forall i, m\)"
}}

***output***:

```
# Add storage capacity constraints
for i in range(I):
    for m in range(M):
        model.addConstr(storage[i, m] <= storageSize[m], name="storage_capacity")
```

Previously you have generated code for defining relevant parameters and variables in the optimization problem. Please refer to the following code when generating the code for this part to make the overall code snippets consistent:

{definition_code}

Here's a constraint we need you to write the code for:

-----
{context}
-----

Take a deep breath and generate the code accordingly.
"""],
    "objective": [
"""
You're an expert programmer who are familiar with optimization problem modeling. Your responsibility is to write Python code for defining objective of the problem.
""", """
- Assume the parameters and variables are defined, and gurobipy is imported as gp. Generate a code accordingly, and don't generate any other text.

Here's an example:

**input**:

{{
    "description": "Maximize the total profit from selling goods",
    "formulation": "Maximize \(Z = \sum_{{k=1}}^{{K}} \sum_{{i=1}}^{{I}} (profit_k \cdot x_{{k,i}} - storeCost \cdot s_{{k,i}})\)"
}}


***output***:

```
# Set objective
model.setObjective(gp.quicksum(profit[k] * x[k, i] - storeCost * s[k, i] for k in range(K) for i in range(I)), gp.GRB.MAXIMIZE)
```

Previously you have generated code for defining relevant parameters and variables in the optimization problem. Please refer to the following code when generating the code for this part to make the overall code snippets consistent:

{definition_code}

Here's the objective function that you are supposed to write the code for:

-----
{context}
-----

Take a deep breath and generate the code accordingly.
""",
    ],
}

debugging_template = """
You're an expert programmer who are familiar with optimization problem modeling. Your responsibility is to debug the code for the problem.

When running the code, a runtime error happened. Please come up with the reason of the runtime error according to the feedback or message of the error, and then fix the code and generate the new code with the following format:

```json
{{
    "reason": "A explanation for the occurrence of the runtime error based on the system traceback information",
    "definition": [
        {{
            "symbol": "The symbol of a problematic variable (definition code should be revised)",
            "code": "A string representing the fixed code for the variable definition"
        }},
        ...
    ],
    "{target}": "A string representing the fixed {target} modeling code to be replaced with the last part code"
}}
```

- There is probably no variable definition code need to be fixed, and in that case, output "definition" list could be empty.
- When defining variables, use model.addVar instead of model.addVars if the variable is a scalar.
- Variables should become before parameters when defining inequality constraints in gurobipy (because of the gurobi parsing order syntax)

Here is the initial part of the code (importing package and defining model):

-----
{prep_code}
-----

Here are the descriptions and dimension information of parameters and variables:

-----
{definition_info}
-----

Here is the definition of paramters and variables:

-----
{definition_code}
-----

The error is because of the this last part which is for modeling the {target}, and here is the context information of {target} that the code attempts to model:

-----
{target_context}
-----

Here is the error line of the code:

-----
{error_line}
-----

and here is the error message:

-----
{error_message}
-----

Take a deep breath and solve the problem step by step.
"""


class Programmer:
    def __init__(
        self, client, solver="gurobipy", model: Optional[str] = "gpt-3.5-turbo"
    ):
        self.client=client
        self.solver = solver
        self.model = model

    def program(self, state: Dict) -> (str, Dict):
        print("Programmer agent is called")
        print()

        if state["sol_status"] == "runtime_error":
            # debugging
            bogus_item = None
            for target in ["constraints", "objective", "variables"]:
                for item in state[target]:
                    if item["status"] == "runtime_error":
                        bogus_item = item
                        break
                if bogus_item is not None:
                    break

            if not bogus_item:
                raise Exception(
                    "No runtime error in state!"
                )

            return self._debugging(state=state, target=target, bogus_item=bogus_item)

        elif state["sol_status"] is None:
            # coding
            return self._coding(state=state)

        else:
            raise Exception(
                f"Invalid solver_output_status {state['solver_output_status']}!"
            )

    def _debugging(self, state: Dict, target:str, bogus_item: Dict) -> (str, Dict):
        error_line = None
        prep_code = state["prep_code"]

        error_line = bogus_item["code"]
        error_message = state["error_message"]

        prompt = debugging_template.format(
            target=target,
            prep_code=prep_code,
            definition_info=json.dumps(self.__get_param_var_def_info(state=state), indent=4),
            definition_code=self.__get_param_var_def_code(state=state),
            error_line=error_line,
            error_message=error_message,
            target_context=json.dumps(
                {
                    "symbol": bogus_item["symbol"],
                    "description": bogus_item["description"],
                    "dim": bogus_item["dim"]
                } if target == "variables" else
                {
                    "description": bogus_item["description"],
                    "formulation": bogus_item["formulation"]
                },
                indent=4
            )
        )
        cnt = 3
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
                response = json.loads(response.replace("```", ""))
                
                for item in response["definition"]:
                    for var in state["variables"]:
                        if var["symbol"] == item["symbol"]:
                            var["code"] = item["code"]
                
                bogus_item["status"] = "coded"
                bogus_item["code"] = response[target]
                return "The code is fixed! Try evaluating it again.", state
            
            except Exception as e:
                print(e)
                cnt -= 1
                if cnt == 0:
                    raise RuntimeError("Debugging failed.")
            
    def _coding(self, state: Dict) -> (str, Dict):
        for parameter in state["parameters"]:
            name = parameter["symbol"]
            if parameter["dim"] == "[]":
                # scalar value
                parameter["code"] = f"{name} = data[\"{parameter['symbol']}\"]"
            else:
                # convert into numpy array
                parameter["code"] = f"{name} = np.array(data[\"{parameter['symbol']}\"])"
            parameter["status"] = "coded"
        
        for variable in state["variables"]:
            print(f"Programming variable {variable['symbol']}")

            messages = [
                {
                    "role": "system",
                    "content": variable_definition_prompt_templates[0]
                },
                {
                    "role": "user",
                    "content": variable_definition_prompt_templates[1].format(
                        variable=json.dumps(variable, indent=4),
                    ),
                },
            ]
            response = model_call(
                client=self.client,
                prompt=messages,
                model=self.model
            )

            code = response.replace("```python", "").replace("```", "").strip()
            print(code)

            variable["code"] = code
            variable["status"] = "coded"
            
        param_var_def_code = self.__get_param_var_def_code(state=state)

        for target in ["constraints", "objective"]:
            for item in state[target]:
                print(f"Programming {target}")

                messages = [
                    {
                        "role": "system",
                        "content": main_prompt_templates[target][0]
                    },
                    {
                        "role": "user",
                        "content": main_prompt_templates[target][1].format(
                            context=json.dumps(item, indent=4),
                            definition_code=param_var_def_code
                        ),
                    },
                ]
                response = model_call(
                    client=self.client,
                    prompt=messages,
                    model=self.model
                )

                code = response.replace("```python", "").replace("```", "").strip()
                print(code)

                item["code"] = code
                item["status"] = "coded"

        return "Coding Done! Now we can evaluate the code!", state

    def __get_param_var_def_code(self, state: Dict) -> str:
        param_var_def_code = ""
        for parameter in state["parameters"]:
            param_var_def_code += parameter["code"] + "\n"
        for variable in state["variables"]:
            param_var_def_code += variable["code"] + "\n"
        return param_var_def_code
    
    def __get_param_var_def_info(self, state: Dict) -> Dict:
        param_var_def_info = dict(parameters=[], variables=[])
        for parameter in state["parameters"]:
            param_var_def_info["parameters"].append({
                "symbol": parameter["symbol"],
                "description": parameter["description"],
                "dim": parameter["dim"]
            })
        for variable in state["variables"]:
            param_var_def_info["variables"].append({
                "symbol": variable["symbol"],
                "description": variable["description"],
                "dim": variable["dim"]
            })
        return param_var_def_info