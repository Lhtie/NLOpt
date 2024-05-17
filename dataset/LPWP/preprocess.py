import os
import json

probs = os.listdir()
for prob in probs:
    if prob.startswith("prob_"):
        with open(f"{prob}/sample.json", "r") as f:
            data = json.load(f)
            
        with open(f"{prob}/input.json", "w") as f:
            json.dump(data[0]["input"], f, indent=4)
            
        with open(f"{prob}/output.json", "w") as f:
            json.dump(data[0]["output"], f, indent=4)