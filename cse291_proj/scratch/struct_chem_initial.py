# import torch
# import csv
# from tqdm import tqdm  # Added for progress tracking

# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     GenerationConfig,
# )

# import re

# from student_dataloader import get_sci_bench_dataloader_for_student
# from generate_student_file_for_scibench import get_model_and_tokenizer
# import prompts.baseline

# # The idea here is to augment normal Chain of Thought reasoning. To do this we will break down Chemistry QnA into two stages: Formula Retrieval and Reasoning.
# # In Formula Retrieval, we will simply ask the model to generate a set of relevant formulae needed to solve the problem.
# # In Reasoning, we will ask the model to generate a set of reasoning steps to solve the problem.
# # Importantly, we will adhere to a Confidence Review and Refinement loop, to ensure that the outputs meet some internal quality threshold.
# # The way this will work is that each time we generate a set of formulae or reasoning steps, we will ask the model to generate a confidence score for each step.
# # We will repeat this process over n_iters and select the generation with the highest confidence score.
# # The Review and Refinement will occur for both formulae and reasoning steps.

# # We'll use a batch size of one, because coordinating variable length reasoning traces along different elements of the batch is something I haven't figured out yet.

# def extract_confidence(text: str) -> float:
#     confidence_match = re.search(r'\*\*Confidence score:\*\*\s*\[([\d.]+)\]', text)
#     return float(confidence_match.group(1)) if confidence_match else 0.0

# def extract_generation_after_prompt(full_output: str, prompt: str) -> str:
#     return full_output[len(prompt):]

# def struct_chem(model, tokenizer, input_ids, attention_mask, num_iters=5):
#     import pdb; pdb.set_trace()

#     # Decode the input IDs to get the initial prompt
#     initial_prompt = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
#     # Generate the first iteration of outputs 
#     initial_generation = model.generate(
#         input_ids=input_ids.squeeze().to("cuda"),
#         attention_mask=attention_mask.squeeze().to("cuda"),
#         pad_token_id=tokenizer.pad_token_id,
#         max_new_tokens=2048,
#         temperature=0.7,         # Controls randomness (0.0-1.0). Lower = more focused
#         top_p=0.9,              # Nucleus sampling threshold. Higher = more diverse
#         top_k=50,               # Limits vocabulary to top K tokens
#         do_sample=True,         # Enable sampling (vs greedy decoding)
#         repetition_penalty=1.1   # Penalize repeated tokens. >1.0 reduces repetition
#     )

#     best_formulae_confidence = 0.0

#     import pdb; pdb.set_trace()

#     # The formulae revising prompt is in prompts.baseline.refine_formulae_prompt
#     for _ in range(num_iters):
#         refined_inputs = tokenizer.decode(generated_ids[0], skip_special_tokens=True) + prompts.baseline.refine_formulae_prompt


# # def main():
# #     model_name = "meta-llama/Llama-3.1-8B-Instruct"

# #     model, tokenizer = get_model_and_tokenizer(model_name)
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     model.eval().to(device)
# #     source = "chemmc"
# #     batch_size = 4
# #     sequence_length = 1024

# #     dataloader = get_sci_bench_dataloader_for_student(tokenizer=tokenizer, source=source, batch_size=batch_size, sequence_length=sequence_length)
# #     print("Dataloader length: ", len(dataloader))

# #     file_path_to_write = "student_generated_cot_text_chemmc.csv"

# #     # Write header and keep file open for the loop
# #     with open(file_path_to_write, 'w', newline='', encoding='utf-8') as f:
# #         writer = csv.writer(f, quoting=csv.QUOTE_ALL)
# #         # Write header
# #         writer.writerow(['question_id', 'generated_text'])
        
# #         # Begin evaluation loop
# #         for batch in tqdm(dataloader, desc="Generating responses"):
# #             input_ids = batch["input_ids"].to(device)
# #             attention_mask = batch["attention_mask"].to(device)
# #             problem_ids = batch["problem_id"].to(device)

# #             # # Generate outputs
# #             # # generated_ids = model.generate(
# #             # #     input_ids=batch["input_ids"].squeeze().to("cuda"),
# #             # #     attention_mask=batch["attention_mask"].squeeze().to("cuda"),
# #             # #     pad_token_id=tokenizer.pad_token_id,
# #             # #     max_new_tokens=512,
# #             # # )
# #             # generated_ids = model.generate(
# #             #     input_ids=batch["input_ids"].squeeze().to("cuda"),
# #             #     attention_mask=batch["attention_mask"].squeeze().to("cuda"),
# #             #     pad_token_id=tokenizer.pad_token_id,
# #             #     max_new_tokens=1024,
# #             #     temperature=0.7,         # Controls randomness (0.0-1.0). Lower = more focused
# #             #     top_p=0.9,              # Nucleus sampling threshold. Higher = more diverse
# #             #     top_k=50,               # Limits vocabulary to top K tokens
# #             #     do_sample=True,         # Enable sampling (vs greedy decoding)
# #             #     repetition_penalty=1.1   # Penalize repeated tokens. >1.0 reduces repetition
# #             # )
# #             generated_ids = struct_chem(model, tokenizer, input_ids, attention_mask)

# #             for i in range(batch_size):
# #                 try:
# #                     question_id = tokenizer.decode(problem_ids[i].squeeze(), skip_special_tokens=True)
# #                     generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
# #                     input_prompt = tokenizer.decode(input_ids[i].squeeze(), skip_special_tokens=True)
                    
# #                     # Clean the text
# #                     question_id = question_id.replace('\n', ' ').replace('\r', ' ').strip()
# #                     generated_text = generated_text.replace('\n', ' ').replace('\r', ' ').strip()
                    
# #                     # Write row
# #                     writer.writerow([question_id, generated_text])
# #                 except IndexError:
# #                     # Handle case where last batch might be smaller
# #                     print(f"Skipping incomplete batch item {i}")
# #                     continue
# #                 except Exception as e:
# #                     print(f"Error processing batch item {i}: {str(e)}")
# #                     continue

# def test_debug_main():
#     model_name = "meta-llama/Llama-3.1-8B-Instruct"

#     model, tokenizer = get_model_and_tokenizer(model_name)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.eval().to(device)
#     source = "chemmc"
#     # batch_size = 1
#     # sequence_length = 1024

#     dataloader = get_sci_bench_dataloader_for_student(tokenizer=tokenizer, source=source)#, batch_size=batch_size, sequence_length=sequence_length)
#     print("Dataloader length: ", len(dataloader))

#     # for batch in tqdm(dataloader, desc="Generating responses"):
#     #         print("Is this code segment being run?")
#     #         input_ids = batch["input_ids"].to(device)
#     #         attention_mask = batch["attention_mask"].to(device)
#     #         problem_ids = batch["problem_id"].to(device)
#     #         import pdb; pdb.set_trace()
#     #         generated_ids = struct_chem(model, tokenizer, input_ids, attention_mask)

#     for batch in dataloader:
#         print("Is this code segment being run?")
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         problem_ids = batch["problem_id"].to(device)

#         generated_ids = struct_chem(model, tokenizer, input_ids, attention_mask)
#         import pdb; pdb.set_trace()

# # debug_main()

