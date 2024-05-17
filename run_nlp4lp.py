import os
import openai
import json
import google.generativeai as genai
from copy import deepcopy
import numpy as np

from frontend import Formulator
from backend import Solver

dataset_path = "dataset/nlp4lp_oct_23_2023"
desc_path = "description.txt"
input_json_path = "data.json"

def run(client, data_path):
    with open(f"{data_path}/{desc_path}", "r") as f:
        raw = f.read().strip()
        obj = raw.split("OBJECTIVE: ")[-1].split("OUTPUT INFO:")[0].strip()
        desc = raw.split("PROBLEM INFO:")[-1].split("INPUT FORMAT:")[0].strip() + "\n" + obj
        input_format = raw.split("INPUT FORMAT:")[-1].split("OBJECTIVE:")[0].strip()
        
    with open(f"{data_path}/{input_json_path}", "r") as f:
        data = json.load(f)
        
    dim_dict = {}
    for line in input_format.split("\n"):
        if ':' in line:
            symbol = line.split(": ")[0].strip()[1:-1]
            form = line.split(": ")[-1].strip()
            dim = []
            for ctx in form.split("]")[:-1]:
                if "," in ctx:
                    dim.append(ctx.split(",")[-1].strip())
            dim.reverse()
            dim_dict[symbol] = f"[{', '.join(dim)}]"
            t = deepcopy(data[symbol])
            while isinstance(t, list):
                data[dim[0]] = len(t)
                dim_dict[dim[0]] = "[]"
                dim = dim[1:]
                if len(t) == 0:
                    break
                t = t[0]
        
    with open(f"{data_path}/input.json", "w") as f:
        json.dump(data, f, indent=4)
    
    param_symbols = [(key, dim_dict[key]) for key in data.keys()]
    print(param_symbols)
    
    frontend = Formulator(
        client=client,
        param_symbols=param_symbols,
        maximum_retries=40
    )
    backend = Solver(
        client=client,
        data_json_path=f"{data_path}/input.json",
        maximum_retries=40
    )
    
    try:
        state = frontend.formulate(desc)
        with open(f"{data_path}/pred_state_40.json", "w") as f:
            json.dump(state, f, indent=4)
        
        res, state = backend.solve(state)
    
    except Exception as e:
        print(e)
        res = "fail"
        
    if res == "success":
        with open(f"{data_path}/pred_sol_40.json", "w") as f:
            json.dump(
                {
                    "obj_val": state["obj_val"],
                    "code": state["code"]
                },
                f, indent=4
            )
        with open("sample.py", "w") as f:
            f.write(state["code"])
        return True
    else:
        print("Failed")
        return False

if __name__ == "__main__":
    with open("google.key") as f:
        key = f.read().strip()
    # client = openai.Client(api_key=key)
    genai.configure(api_key=key, transport='rest')
    client = genai.GenerativeModel('gemini-pro')
    
    brenchs = os.listdir(dataset_path)
    cnt, re = 0, 0
    for brench in brenchs:
        probs = os.listdir(os.path.join(dataset_path, brench))
        count = []
        for prob in probs:
            # if brench != "introduction_to_linear_optimization" or prob != "problem_1":
            #     continue
            prob_path = f"{dataset_path}/{brench}/{prob}"
            if os.path.isdir(prob_path) and \
                    desc_path in os.listdir(prob_path) and \
                    input_json_path in os.listdir(prob_path) and \
                    "obj.txt" in os.listdir(prob_path):
                
                print(f"Testing {brench}/{prob}")
                with open(f"{prob_path}/obj.txt", "r") as f:
                    gt = float(f.read().strip()[5:])
                succ = run(client, prob_path)
                
                cnt += 1
                if not succ:
                    re += 1
                
    print(cnt, re)
