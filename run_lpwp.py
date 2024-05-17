import os
import openai
import json
import google.generativeai as genai
from copy import deepcopy
import numpy as np

from frontend import Formulator
from backend import Solver

dataset_path = "dataset/LPWP"
desc_path = "description.txt"
input_json_path = "input.json"

def run(client, data_path):
    with open(f"{data_path}/{desc_path}", "r") as f:
        desc = f.read().strip()
        
    with open(f"{data_path}/{input_json_path}", "r") as f:
        data = json.load(f)
    
    param_symbols = [(key, "[]") for key in data.keys()]
    print(param_symbols)
    
    frontend = Formulator(
        client=client,
        param_symbols=param_symbols,
        maximum_retries=10
    )
    backend = Solver(
        client=client,
        data_json_path=f"{data_path}/{input_json_path}",
        maximum_retries=10
    )
    
    try:
        state = frontend.formulate(desc)
        with open(f"{data_path}/pred_state_10.json", "w") as f:
            json.dump(state, f, indent=4)
        
        res, state = backend.solve(state)
    
    except Exception as e:
        print(e)
        res = "fail"
        
    if res == "success":
        with open(f"{data_path}/pred_sol_10.json", "w") as f:
            json.dump(
                {
                    "obj_val": state["obj_val"],
                    "code": state["code"]
                },
                f, indent=4
            )
        with open("sample.py", "w") as f:
            f.write(state["code"])
        return True, state["obj_val"]
    else:
        print("Failed")
        return False, None

if __name__ == "__main__":
    with open("google.key") as f:
        key = f.read().strip()
    # client = openai.Client(api_key=key)
    genai.configure(api_key=key, transport='rest')
    client = genai.GenerativeModel('gemini-pro')
    
    probs = os.listdir(dataset_path)
    cnt, re, corr = 0, 0, 0

    count = []
    for prob in probs:
        prob_path = f"{dataset_path}/{prob}"
        if os.path.isdir(prob_path) and \
                desc_path in os.listdir(prob_path) and \
                input_json_path in os.listdir(prob_path) and \
                "output.json" in os.listdir(prob_path):
                    
            print(f"Testing {prob}")
            with open(f"{prob_path}/output.json", "r") as f:
                gt = json.load(f)[0]
            succ, obj_val = run(client, prob_path)
            
            cnt += 1
            if not succ:
                re += 1
            elif not isinstance(obj_val, str) and abs(gt - obj_val) < 1:
                corr += 1
            
                
    print("Total samples:", cnt)
    print("Runtime error samples:", re)
    print("Accepted samples:", corr)
