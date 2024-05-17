import os
import openai
import json
import google.generativeai as genai
from copy import deepcopy

from utils.ModelCall import model_call

dataset_path = "dataset/nlp4lp_oct_23_2023"
desc_path = "description.txt"
input_json_path = "data.json"

prompt = """
You are a Python programmer in the field of operations research and optimization.

Your proficiency in utilizing third-party libraries such as Gurobi is essential. In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, or PuLP. 

You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem. 

Now the origin problem is as follow:

{problem}

- The input format is provided above. In your code, please read the inputs from JSON file {data_file}.
- Make sure that the final optimal objective value is stored in a variable named \"obj_val\".

Give your Python code directly.
"""

def run(client, data_path):
    with open(f"{data_path}/{desc_path}", "r") as f:
        desc = f.read().strip()
        
    with open(f"{data_path}/{input_json_path}", "r") as f:
        data = json.load(f)

    try:
        response = model_call(
            client=client,
            prompt=prompt.format(problem=desc, data_file=f"{data_path}/{input_json_path}")
        )
        code = response.replace("```python", "").replace("```", "").strip()
        print(code)
        
        local_env = {}
        exec(
                code,
                local_env,
                local_env,
            )
        obj_val = local_env["obj_val"]
        res = "success"
    
    except Exception as e:
        print(e)
        res = "fail"
        
    if res == "success":
        with open(f"{data_path}/standard.json", "w") as f:
            json.dump(
                obj_val,
                f, indent=4
            )
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
    
    probs = os.listdir(dataset_path)
    cnt, re, corr = 0, 0, 0

    brenchs = os.listdir(dataset_path)
    cnt, re = 0, 0
    for brench in brenchs:
        probs = os.listdir(os.path.join(dataset_path, brench))
        for prob in probs:
            if brench == "introduction_to_linear_optimization" and prob == "problem_16":
                flag = True
            if not flag:
                continue
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
                
    print("Total samples:", cnt)
    print("Runtime error samples:", re)
