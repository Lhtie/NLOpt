import json
import traceback

from typing import Dict, Optional

prep_code = """
import json
import numpy as np
import math

{solver_prep_code}

with open("{data_json_path}", "r") as f:
    data = json.load(f)
"""


post_code = """

# Get model status
status = model.status

obj_val = None
# check whether the model is infeasible, has infinite solutions, or has an optimal solution
if status == gp.GRB.INFEASIBLE:
    obj_val = "infeasible"
elif status == gp.GRB.INF_OR_UNBD:
    obj_val = "infeasible or unbounded"
elif status == gp.GRB.UNBOUNDED:
    obj_val = "unbounded"
elif status == gp.GRB.OPTIMAL:
    obj_val = model.objVal
"""


class Evaluator:
    def __init__(self, client, solver="gurobipy", model: Optional[str] = "gpt-3.5-turbo"):
        self.client = client
        self.solver = solver
        self.model = model

    def eval(self, state: Dict) -> (str, Dict):
        print("Evaluator agent is called")

        res = self._run(state=state)

        if not res["success"]:
            state["sol_status"] = "runtime_error"
            state["solver_output_status"] = "runtime_error"
            state["error_message"] = res["error_message"]
            state["prep_code"] = prep_code.format(
                solver_prep_code=self.get_solver_prep_code(),
                data_json_path=state["data_json_path"],
            )

            if not res["bogus_context"]:
                return f"Bad model! Print DONE to finish the execution. Error_msg: {res['error_message']}", state
            res["bogus_context"]["status"] = "runtime_error"

            return (
                f"There was an error in running the code! {res['error_message']}",
                state,
            )

        else:
            state["sol_status"] = "solved"
            state["solver_output_status"] = res["status"]
            state["obj_val"] = res["obj_val"]
            state["code"] = res["code"]
            return ("Evaluation Done! The problem is solved.", state)

    def _run(self, state: Dict):
        local_env = {}
        code = ""
        last_line = ""
        bogus_context = None

        try:
            last_line = prep_code.format(
                solver_prep_code=self.get_solver_prep_code(),
                data_json_path=state["data_json_path"],
            )
            for parameter in state["parameters"]:
                last_line += parameter["code"] + "\n"
            code += last_line

            exec(
                last_line,
                local_env,
                local_env,
            )

            for variable in state["variables"]:
                bogus_context = variable
                last_line = variable["code"]
                code += last_line + "\n"
                exec(last_line, local_env, local_env)

            for constraint in state["constraints"]:
                bogus_context = constraint
                last_line = constraint["code"]
                code += "\n" + last_line + "\n"
                exec(last_line, local_env, local_env)

            bogus_context = state["objective"][0]
            last_line = state["objective"][0]["code"]
            code += "\n" + last_line + "\n"
            exec(last_line, local_env, local_env)

            bogus_context = "OPTIMIZATION CALL"
            last_line = f"\n# Optimize model\nmodel.optimize()\n"
            code += last_line + "\n"
            exec(last_line, local_env, local_env)

            bogus_context = None
            last_line = post_code
            code += last_line + "\n"
            exec(last_line, local_env, local_env)

            return {
                "success": True,
                "error_line": None,
                "code": code,
                "obj_val": local_env["obj_val"],
                "status": local_env["status"],
                "error_message": None
            }

        except Exception as e:
            print("RUNTIME ERROR")
            print(code)

            error_msg = traceback.format_exc()
            return {
                "success": False,
                "error_line": last_line,
                "code": code,
                "obj_val": None,
                "status": None,
                "error_message": error_msg,
                "bogus_context": bogus_context
            }

    def get_solver_prep_code(self):
        if self.solver == "gurobipy":
            return "import gurobipy as gp\n\n # Define model\nmodel = gp.Model('model')"
        else:
            raise Exception(f"Solver {self.solver} is not supported yet!")
