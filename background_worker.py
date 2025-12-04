import sys, json, os
from mwp_classes import PromptGenerator, ProblemEvaluator, MWPGenerator

session_file = sys.argv[1]
output_folder = sys.argv[2]

with open(session_file) as f:
    session_data = json.load(f)

api_key = session_data["api_key"][0]
curr_dl = session_data["curr_dl"]
curr_text_l = session_data["curr_text_l"]
ex_dat_path = session_data["ex_dat_path"]
standards_path = session_data["standards_path"]
existing_queue = session_data.get("existing_queue", [])

with open(ex_dat_path) as f:
    ex_dat = json.load(f)
with open(standards_path) as f:
    standards_dat = json.load(f)

generator = MWPGenerator(api_key=api_key, target_dl=curr_dl, target_text_l=curr_text_l)

# generate until we have enough problems
needed = 5 - len(existing_queue)
new_problems = []

while len(new_problems) < needed:
    candidates = generator.generate_item(
        ex_dat=ex_dat, 
        standards_dat=standards_dat, 
        n_examples=2, 
        n_items=10, 
        model_option="gpt-4.1-mini", 
        rationale_dat=True
    )
    generator.evaluator.problem_db = existing_queue
    new_problems.extend(generator.evaluator.parallel_evaluation(candidates))
    

# save each problem separately for Streamlit to pick up
os.makedirs(output_folder, exist_ok=True)
for i, p in enumerate(new_problems):
    with open(os.path.join(output_folder, f"problem_{i}.json"), "w") as f:
        json.dump(p, f)
