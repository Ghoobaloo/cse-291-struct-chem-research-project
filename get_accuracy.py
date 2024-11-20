import json
import math
from post_process import parse_math_answer, remove_not, cal_not,parse_not

def equiv(model_output, answer, unit):
    model_output=model_output.replace(',', '')
    try:
        first=math.isclose(float(model_output.strip()), float(answer.strip()), abs_tol=0.1)
    except:
        first=False
    try: 
        model=model_output.strip().split()[0]
        second=math.isclose(float(model), float(answer.strip()), abs_tol=0.1)
    except:
        second=False
    if first or second:
        return True
    return False

#DATASET = "atkins"
#SIZE = 105
#DATASET = "chemmc"
#SIZE = 38
#DATASET = "matter"
#SIZE = 47
DATASET = "quan"
SIZE = 33

with open("./scibench/dataset/original/" + DATASET + ".json") as f:
    original = json.load(f)

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

ans = load_jsonl("./outputs/" + DATASET + "_res.jsonl")[-SIZE:]

print(len(original), len(ans))

assert len(ans) == len(original)

correct = 0
for i in range(len(ans)):

    problem_data = original[i]
    model_output_ori = ans[i]['gpt_output']

    #print("model output: ", model_output_ori)

    unit_prob=problem_data["unit"]
    if remove_not(problem_data["unit"]):
        unit_prob=remove_not(problem_data["unit"])

    model_output = parse_math_answer(model_output_ori)
    answer = problem_data["answer_number"]
    # if unit_prob != problem_data["unit"]:
    #     model_output=cal_not(parse_not(model_output))
    #     answer=cal_not((answer, problem_data["unit"]))

    # print(model_output)
    #print("answer: ", answer)
    #input()

    try:
        res_equiv = equiv(model_output, answer, problem_data["unit"])
    except:
        res_equiv = False
    if res_equiv:
        correct += 1

    # print(res_equiv)
    # input()

print(DATASET, correct/len(ans))
