import torch
import csv
from tqdm import tqdm  # Added for progress tracking

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

import re

# from student_dataloader import get_sci_bench_dataloader_for_student
# from generate_student_file_for_scibench import get_model_and_tokenizer
import prompts.prompts_final

# The idea here is to augment normal Chain of Thought reasoning. To do this we will break down Chemistry QnA into two stages: Formula Retrieval and Reasoning.
# In Formula Retrieval, we will simply ask the model to generate a set of relevant formulae needed to solve the problem.
# In Reasoning, we will ask the model to generate a set of reasoning steps to solve the problem.
# Importantly, we will adhere to a Confidence Review and Refinement loop, to ensure that the outputs meet some internal quality threshold.
# The way this will work is that each time we generate a set of formulae or reasoning steps, we will ask the model to generate a confidence score for each step.
# We will repeat this process over n_iters and select the generation with the highest confidence score.
# The Review and Refinement will occur for both formulae and reasoning steps.

# We'll use a batch size of one, because coordinating variable length reasoning traces along different elements of the batch is something I haven't figured out yet.

def extract_confidence(text: str) -> int:
    """Extract confidence score from various possible formats:
    - **Confidence score:** [95]
    - **Confidence score:** 95
    - **Confidence Score**: 95
    - **Confidence Score**: \n95
    - **Confidence Score**:\n95
    - **Confidence Score**: \n95
    - **Confidence Score**: \n[95]
    """
    confidence_match = re.search(
        r'\*\*Confidence [Ss]core\*\*:[ \t]*\n*[ \t]*(?:\[?(\d+)\]?)',
        text,
        re.IGNORECASE
    )
    return int(confidence_match.group(1)) if confidence_match else 0

def extract_generation_after_prompt(full_output: str, prompt: str) -> str:
    return full_output[len(prompt):]

def struct_chem_beam(model, tokenizer, device, input_ids, attention_mask, num_beams=3):
    """
    Structured chemistry solver using HuggingFace's built-in beam search for both stages.
    """
    initial_prompt = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    
    # Stage 1: Formula Retrieval with beam search
    formula_generation = model.generate(
        input_ids=input_ids.squeeze(1).to(device),
        attention_mask=attention_mask.squeeze(1).to(device),
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=2048,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        temperature=0.7,
        top_p=0.5,
        top_k=50,
    )

    #import pdb; pdb.set_trace()
    
    # Process all beam outputs and select highest confidence
    formula_candidates = []
    for beam in formula_generation:
        full_text = tokenizer.decode(beam, skip_special_tokens=True)
        generation_text = extract_generation_after_prompt(full_text, initial_prompt)
        confidence = extract_confidence(generation_text)
        formula_candidates.append((generation_text, confidence))
    
    best_formula = max(formula_candidates, key=lambda x: x[1])[0]
    
    # Stage 2: Reasoning with beam search
    reasoning_prompt = initial_prompt + best_formula + prompts.prompts_final.refine_reasoning_prompt_optimized
    tokenized_reasoning_prompt = tokenizer.__call__(
        reasoning_prompt,
        max_length=2048,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    reasoning_generation = model.generate(
        input_ids=tokenized_reasoning_prompt["input_ids"].to(device),
        attention_mask=tokenized_reasoning_prompt["attention_mask"].to(device),
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=2048,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        temperature=0.7,
        top_p=0.5,
        top_k=50,
    )

    #import pdb; pdb.set_trace()
    
    # Process all beam outputs and select highest confidence
    reasoning_candidates = []
    for beam in reasoning_generation:
        full_text = tokenizer.decode(beam, skip_special_tokens=True)
        generation_text = extract_generation_after_prompt(full_text, reasoning_prompt)
        confidence = extract_confidence(generation_text)
        reasoning_candidates.append((generation_text, confidence))

    #import pdb; pdb.set_trace()
    
    best_reasoning = max(reasoning_candidates, key=lambda x: x[1])[0]
    
    return best_reasoning

def struct_chem_original(model, tokenizer, device, input_ids, attention_mask, num_iters=2):

    # Decode the input IDs to get the initial prompt
    initial_prompt = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    # Generate the first iteration of outputs 
    initial_generation = model.generate(
        input_ids=input_ids.squeeze(1).to(device),
        attention_mask=attention_mask.squeeze(1).to(device),
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=2048,
        temperature=0.7,         # Controls randomness (0.0-1.0). Lower = more focused
        top_p=0.9,              # Nucleus sampling threshold. Higher = more diverse
        top_k=50,               # Limits vocabulary to top K tokens
        do_sample=True,         # Enable sampling (vs greedy decoding)
        repetition_penalty=1.1   # Penalize repeated tokens. >1.0 reduces repetition
    )


    initial_full_generation_text = tokenizer.decode(initial_generation[0], skip_special_tokens=True)
    initial_generation_text = extract_generation_after_prompt(initial_full_generation_text, initial_prompt)
    # refined_formulae_generation_prompt = initial_generation_text + prompts.baseline.refine_formulae_prompt
    best_formulae_confidence = extract_confidence(initial_generation_text)
    best_formulae_generation = initial_generation_text
    most_recent_formulae_generation = initial_generation_text

    # # Tokenize the refined_formulae_generation
    # tokenized_refined_formulae_generation = tokenizer.__call__(
    #     refined_formulae_generation_prompt,
    #     max_length=2048,
    #     padding="max_length",
    #     truncation=True,
    #     return_tensors="pt",
    # )

    # The formulae revising prompt is in prompts.baseline.refine_formulae_prompt
    for _ in range(num_iters - 1):

        next_generation = model.generate(
            input_ids=input_ids.squeeze(1).to(device),
            attention_mask=attention_mask.squeeze(1).to(device),
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=2048,
            temperature=0.7,         # Controls randomness (0.0-1.0). Lower = more focused
            top_p=0.9,              # Nucleus sampling threshold. Higher = more diverse
            top_k=50,               # Limits vocabulary to top K tokens
            do_sample=True,         # Enable sampling (vs greedy decoding)
            repetition_penalty=1.1   # Penalize repeated tokens. >1.0 reduces repetition
        )

        next_full_generation_text = tokenizer.decode(next_generation[0], skip_special_tokens=True)
        next_generation_text = extract_generation_after_prompt(next_full_generation_text, initial_prompt)
        # refined_formulae_generation_prompt = initial_generation_text + prompts.baseline.refine_formulae_prompt
        new_formulae_confidence = extract_confidence(next_generation_text)
        most_recent_formulae_generation = next_generation_text
        

        if new_formulae_confidence >= best_formulae_confidence:
            best_formulae_confidence = new_formulae_confidence
            best_formulae_generation = next_generation_text

    # We repeat the process for reasoning refinement
    new_prompt_with_best_formulae = initial_prompt + best_formulae_generation + prompts.prompts_final.refine_reasoning_prompt
    tokenized_new_prompt = tokenizer.__call__(
        new_prompt_with_best_formulae,
        max_length=2048,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized_new_prompt["input_ids"]
    attention_mask = tokenized_new_prompt["attention_mask"]

    next_generation = model.generate(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=2048,
        temperature=0.7,         # Controls randomness (0.0-1.0). Lower = more focused
        top_p=0.9,              # Nucleus sampling threshold. Higher = more diverse
        top_k=50,               # Limits vocabulary to top K tokens
        do_sample=True,         # Enable sampling (vs greedy decoding)
        repetition_penalty=1.1   # Penalize repeated tokens. >1.0 reduces repetition
    )

    next_full_generation_text = tokenizer.decode(next_generation[0], skip_special_tokens=True)
    next_generation_text = extract_generation_after_prompt(next_full_generation_text, new_prompt_with_best_formulae)
    best_reasoning_confidence = extract_confidence(next_generation_text)
    best_reasoning_generation = next_generation_text
    most_recent_reasoning_generation = next_generation_text

    for _ in range(num_iters - 1):
        next_generation = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=2048,
            temperature=0.7,         # Controls randomness (0.0-1.0). Lower = more focused
            top_p=0.9,              # Nucleus sampling threshold. Higher = more diverse
            top_k=50,               # Limits vocabulary to top K tokens
            do_sample=True,         # Enable sampling (vs greedy decoding)
            repetition_penalty=1.1   # Penalize repeated tokens. >1.0 reduces repetition
        )

        next_full_generation_text = tokenizer.decode(next_generation[0], skip_special_tokens=True)
        next_generation_text = extract_generation_after_prompt(next_full_generation_text, new_prompt_with_best_formulae)
        new_reasoning_confidence = extract_confidence(next_generation_text)
        most_recent_reasoning_generation = next_generation_text

        if new_reasoning_confidence >= best_reasoning_confidence:
            best_reasoning_confidence = new_reasoning_confidence
            best_reasoning_generation = next_generation_text

    return best_reasoning_generation
        

# def test_debug_main():
#     model_name = "meta-llama/Llama-3.1-8B-Instruct"

#     model, tokenizer = get_model_and_tokenizer(model_name)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.eval().to(device)
#     source = "chemmc"

#     dataloader = get_sci_bench_dataloader_for_student(tokenizer=tokenizer, source=source)#, batch_size=batch_size, sequence_length=sequence_length)
#     print("Dataloader length: ", len(dataloader))


#     for batch in tqdm(dataloader, desc="Generating responses"):
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         problem_ids = batch["problem_id"].to(device)

#         response = struct_chem(model, tokenizer, device, input_ids, attention_mask)

# test_debug_main()

