from datetime import datetime, timedelta
import logging
import time
import json
import random
import re
import io
import sys
from prompts.instruction import overall_instruction, overall_instruction_pot
import argparse
import os
from prompts.refine_prompt import refine_formulae_prompt, refine_reasoning_prompt, refine_reasoning_prompt_pot
# from azure.identity import DefaultAzureCredential
import openai
from openai import OpenAI
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel
from huggingface_hub import hf_hub_download
import ollama

HUGGINGFACE_API_KEY = "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"

#model_id = "google/gemma-2-2b-it"
#filenames = ["config.json", "generation_config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json"]

#model_id = "google/gemma-2-9b-it"
#filenames = ["config.json", "generation_config.json", "model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors", "model.safetensors.index.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json"]

#model_id_tok = model_id

#model_id = "SanctumAI/gemma-2-9b-it-GGUF"
#filenames = ["config.json", "gemma-2-9b-it.Q4_1.gguf"]
#gguf_filename = "gemma-2-9b-it.Q4_K.gguf"
#model_path = "SanctumAI/gemma"

model_name = "hf.co/SanctumAI/gemma-2-9b-it-GGUF:Q4_K_M"
#response = ollama.generate(model=model_name, options={'temperature': 0.0}, prompt="Give me a success score for this execution from 1 to 10. Respond with a single number")
#print(response)
#print(response['response'])


device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device: ', device)
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())
#print(refine_reasoning_prompt_pot)

#for filename in filenames:
#    downloaded_model_path = hf_hub_download(
#        repo_id = model_id,
#        filename = filename,
#        token = HUGGINGFACE_API_KEY
#    )
#    print(downloaded_model_path)


def load_prompt(file):
    prompt = ''
    with open(file) as f:
        for line in f:
            prompt = prompt + line.strip() + '\n'
    return prompt

class HuggingFaceModel:
    
    def __init__(self, max_tokens=1024, temperature=0.0, logprobs=None, n=1, engine='gpt-4',
        frequency_penalty=0, presence_penalty=0, stop=None, rstrip=False, **kwargs):
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        #for gguf attempt
        #self.tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=gguf_filename)
        #self.model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=gguf_filename)

        #self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        #self.model = AutoModelForCausalLM.from_pretrained(model_id)

        #self.pipeline = pipeline("text-generation", model=self.model, device=device, do_sample=False, temperature=temperature, tokenizer=self.tokenizer, truncation=True, max_length=max_tokens)

    def complete(self, prompt):

        #pipeline takes only a string for this model so would need to switch to a chat model later
        messages = [{"role": "system",
                     "content": "You are an expert chemist. Your expertise lies in reasoning and addressing chemistry problems. "},
                     {"role": "user",
                      "content": prompt,}]
        #not perfect, could use some work later
        message = "System: You are an expert chemist. Your expertise lies in reasoning and addressing chemistry problems. User: " + prompt

        print('STARTED GENERATION') 
        #print('the prompt is: ', message)
        #response = self.pipeline(message)
        #answer = response[0]['generated_text']
        #answer = answer[len(message):]

        response = ollama.generate(model=model_name, options={'temperature': 0.0}, prompt=message)
        answer = response['response']

        print('ENDED GENERATION')
        #print('got response: ', answer)

        return answer

class GPT4:

    def __init__(self, max_tokens=1024, temperature=0.0, logprobs=None, n=1, engine='gpt-4',
        frequency_penalty=0, presence_penalty=0, stop=None, rstrip=False, **kwargs):

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rstrip = rstrip
        self.engine = engine
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", None),
        )

    def complete(self, prompt):

        openai.api_version = '2023-03-15-preview'
        self.deployment_id = self.engine
        
        if self.rstrip:
            # Remove heading whitespaces improves generation stability. It is
            # disabled by default to keep consistency.
            prompt = prompt.rstrip()
        retry_interval_exp = 1 

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_id,
                    messages=[
                        {"role": "system", "content": "You are an expert chemist. Your expertise lies in reasoning and addressing chemistry problems. "},
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature = self.temperature,
                    max_tokens = self.max_tokens,
                )
                return response.choices[0].message.content
            
            except openai.RateLimitError as e:
                # NOTE: openai.error.RateLimitError: Requests to the
                # Deployments_Completion Operation under OpenAI API have
                # exceeded rate limit of your current OpenAI S0 pricing tier.
                # Please retry after 7 seconds. Please contact Azure support
                # service if you would like to further increase the default rate
                # limit.
                logging.warning("OpenAI rate limit error. Retry")
                # Expontial backoff
                time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                retry_interval_exp += 1

            except openai.APIConnectionError as e:
                logging.warning("OpenAI API connection error. Retry")
                time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                retry_interval_exp += 1

            # except openai.Timeout as e:
            #     logging.warning("OpenAI timeout error. Sleep then retry.")
            #     time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
            #     retry_interval_exp += 1

MODEL = HuggingFaceModel()

def verify_formula(problem_statement: str, formulae: str, max_attempts: int) -> str:

    #added by me
    global refine_formulae_prompt

    gpt = MODEL
    formulae_retrieved = formulae

    def is_refinement_sufficient(prompt, feedback, initial, refined) -> bool:
        # Define stopping criteria here
        pass

    flag = True

    n_attempts = 0
    max_confidence = 0.0
    
    while n_attempts < max_attempts:
        
        with io.StringIO() as f:
            f.write(refine_formulae_prompt.strip() + "\n\n")
            f.write("Now try the following. Remember to strictly follow the output format:\n\n")
            f.write(f"### Chemistry problem:###\n {problem_statement}\n\n### Formulae retrieval:###\n{formulae_retrieved}")
            model_input = f.getvalue()

        refined_formulae = gpt.complete(model_input)
        
        print('WE HAVE INPUT REFINE FORMULA: ', model_input)
        print('WE HAVE REFINED FORMULA: ', refined_formulae)

        refined_formulae = '**Judgement of the retrieved formulae:**' + refined_formulae.split('**Judgement of the retrieved formulae:**')[1].strip()

        print('DEBUG: ', refined_formulae)
        print('SPLIT: ', len(refined_formulae.split('**Confidence score:**')))

        formulae_new, conf_f = refined_formulae.split('**Confidence score:**')[0].strip("\n"), refined_formulae.split('**Confidence score:**')[1].strip()
        conf_f = conf_f.splitlines()[0]
        
        print('NEW FORMULA WOO: ', formulae_new)
        # extract the confidence score and the refined components
        conf = float(re.findall(r"\d+\.?\d*", conf_f)[0])
        formulae_new = "**Formula retrieval:**" + formulae_new.split("**Formula retrieval:**")[1]

        print('WE HAVE FORMULA NEW: ', formulae_new)
        print('WE HAVE CONFIDENCE SCORE: ', conf)

        if conf > max_confidence:
            max_confidence = conf
            formulae = formulae_new
        else:
            formulae = formulae

        n_attempts += 1

    if n_attempts > 0 :
        flag = False

    return formulae, flag

def verify_reasoning(problem_statement: str, formula: str, reasoning: str, max_attempts: int, pot: bool) -> str:
    
    #added by me
    global refine_reasoning_prompt

    gpt = MODEL

    def is_refinement_sufficient(prompt, feedback, initial, refined) -> bool:
        # Define stopping criteria here
        pass

    flag = True

    n_attempts = 0
    max_confidence = 0.0

    if pot:
        refine_reasoning_prompt = refine_formulae_prompt_pot

    while n_attempts < max_attempts:

        with io.StringIO() as f:
            f.write(refine_reasoning_prompt.strip() + "\n\n")
            f.write("Now try the following. Remember to strictly follow the output format:\n\n")
            f.write(f"### Chemistry problem:###\n {problem_statement}\n\n### Formulae retrieval:###\n{formula}\n\n###Reasoning process###\n{reasoning}")
            model_input = f.getvalue()
        
        refined_reasoning = gpt.complete(model_input)

        print('GOT REASONING INPUT: ', model_input)
        print('GOT REFINED REASONING: ', refined_reasoning)

        refined_reasoning = '**Judgement of the reasoning process:**' + refined_reasoning.split('**Judgement of the reasoning process:**')[1].strip()

        reasoning_new, conf_f = refined_reasoning.split("**Confidence score:**")[0].strip("\n"), refined_reasoning.split("**Confidence score:**")[1].strip()
        conf_f = conf_f.splitlines()[0]

        print('GOT REASONING NEW: ', reasoning_new)
        print('GOT CONFIDENCE STRING: ', conf_f)

        # extract the confidence score and the refined components
        conf = float(re.findall(r"\d+\.?\d*", conf_f)[0])
        reasoning_new = "**Reasoning/calculation process:**" + reasoning_new.split("**Reasoning/calculation process:**")[1]

        print('GOT REASONING NEW TWO: ', reasoning_new)
        print('GOT CONFIDENCE SCORE: ', conf)

        if conf > max_confidence:
            max_confidence = conf
            reasoning = reasoning_new
        else:
            reasoning = reasoning

        n_attempts += 1

    if n_attempts > 0 :
        flag = False

    return reasoning, flag


def run(file, max_attempts, base_lm, mode, pot):

    gpt4 = MODEL
    if pot:
        prompt = overall_instruction_pot
    else:
        prompt = overall_instruction
    # prompt = load_prompt("./prompts/instruction.txt")

    with open("./datasets/{}.json".format(file)) as f:
        test_data = json.load(f)

    nr = 0
    for item in tqdm(test_data):
        nr += 1
        print('DOING: ', nr, ' OUT OF ', len(tqdm(test_data)))

        problem_text = item['problem_text']
        unit_prob = item['unit']
        new_problem = "\n\n Now try to solve the following problem:\n" + problem_text + " The unit of the answer is " + unit_prob + "."
        
        if mode == 'zero-shot':
            problem_statement = prompt + new_problem
        elif mode == 'few-shot':
            # Randomly select three demonstrations
            txt_files = [file for file in os.listdir("./prompts/demonstrations") if file.endswith('.txt')]
            random_files = random.sample(txt_files, 3)
            demonstrations = "" 
            for demo in random_files:
                demo_content = load_prompt(os.path.join("./prompts/demonstrations", demo)) + "\n\n"
                demonstrations += demo_content
            problem_statement = prompt + "\n\nTo clearly explain the task, we provide the following example:" + demonstrations + new_problem
        
        ### 1. First decompose the problem solving process with formulae retrieval and reasoning process
        response = gpt4.complete(prompt=problem_statement)

        print('GOT FIRST RESPONSE: ', response, '\n\n')

        ## 1.1 Parse the generated results of formulae and reasoning
        formula_retrieval, reasoning_process = response.split("**Reasoning/calculation process:**")[0], response.split("**Reasoning/calculation process:**")[1]
        reasoning_process = "**Reasoning/calculation process:**" + reasoning_process.split("**Answer conclusion:**")[0]

        print('GOT FORMULAS: ', formula_retrieval, '\n\n')
        print('GOT REASONING: ', reasoning_process, '\n\n')
        
        ### 2. Iterative review and refinement of formulae and reasoning
        feedback_problem = problem_text + " The unit of the answer is " + unit_prob + "."
        formula_refined, flag_formula = verify_formula(feedback_problem, formula_retrieval, max_attempts)
        reasoning_refined, flag_reasoning = verify_reasoning(feedback_problem, formula_refined, reasoning_process, max_attempts, pot)

        print('AFTER REFINEMENT WE HAVE FORMULA: ', formula_refined, '\n\n')
        print('AFTER REFINEMENT WE HAVE REASONING: ', reasoning_refined, '\n\n')
        print('WE HAVE FLAGS: ', flag_formula, flag_reasoning, '\n\n')

        ### 3. Conclude the answers
        if not pot:
            if flag_formula and flag_reasoning:
                final_response = response
            else:
                verified_prompt = load_prompt("./prompts/verified_instruction.txt")
                print('VERIFIED PROMPT: ', verified_prompt+formula_refined+reasoning_refined)
                final_response = gpt4.complete(prompt=verified_prompt+formula_refined+reasoning_refined)
        else:
            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()
            try:
                reasoning_pot = reasoning_refined.split("**Reasoning/calculation process:**")[1]
                exec(reasoning_pot)
                sys.stdout = old_stdout
                final_response = redirected_output.getvalue().strip()
            except:
                final_response = "None"

        print('LASTLY THE FINAL RESPONSE: ', final_response, '\n\n')

        cur = {}
        cur['gpt_output'] = final_response

        with open('./outputs/{}_res.jsonl'.format(file), 'a') as f:

            print(json.dumps(cur, ensure_ascii=False), file=f)
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4')
    parser.add_argument('--refine_iteration', type=int, default=5)     
    parser.add_argument('--dataset', nargs='+', default=["atkins", "chemmc","matter","quan"])
    parser.add_argument('--mode', type=str, default='zero-shot')
    parser.add_argument('--pot', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    for f in args.dataset:
        run(f, max_attempts=args.refine_iteration, base_lm=args.engine, mode=args.mode, pot=args.pot)
