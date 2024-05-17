import os
import json

dataset_path = "dataset/LPWP"
desc_path = "description.txt"
input_json_path = "input.json"
output_json_path = "output.json"

probs = os.listdir(dataset_path)
cnt, me, re, wa, corr = 0, 0, 0, 0, 0
for prob in probs:
    prob_path = f"{dataset_path}/{prob}"
    if os.path.isdir(prob_path) and \
            desc_path in os.listdir(prob_path) and \
            input_json_path in os.listdir(prob_path) and \
            output_json_path in os.listdir(prob_path):
        with open(f"{prob_path}/{output_json_path}", "r") as f:
            gt = json.load(f)[0]
        if not "pred_state.json" in os.listdir(prob_path):
            me += 1
        elif not "pred_sol.json" in os.listdir(prob_path):
            re += 1
        else:
            with open(f"{prob_path}/pred_sol.json", "r") as f:
                sol = json.load(f)
            obj_val = sol["obj_val"]
            if not isinstance(obj_val, str) and abs(gt - obj_val) < 1:
                corr += 1
            else:
                wa += 1
                print(prob_path, gt, obj_val)
        cnt += 1
            
print("Total samples:", cnt)
print("Modeling error samples:", me)
print("Runtime error samples:", re)
print("Wrong answer samples:", wa)
print("Accepted samples:", corr)