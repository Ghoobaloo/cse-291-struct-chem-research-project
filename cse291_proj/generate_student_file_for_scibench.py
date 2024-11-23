import torch
import csv
from tqdm import tqdm  # Added for progress tracking

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

from student_dataloader import get_sci_bench_dataloader_for_student

def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def verify_dataloader_uniqueness(dataloader):
    seen_samples = set()
    total_samples = 0

    for batch in dataloader:
        for sample in batch["input_ids"]:
            sample_tuple = tuple(sample)
            #import pdb; pdb.set_trace()
            if sample_tuple in seen_samples:
                print("Duplicate sample found!")
                return False
            seen_samples.add(sample_tuple)
            total_samples += 1

    print(f"No duplicates found in {total_samples} samples")
    return True

# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a16"
# model_name = "SanctumAI/gemma-2-9b-it-GGUF:Q4_K_M"

model, tokenizer = get_model_and_tokenizer(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(device)
source = "chemmc"
batch_size = 4
sequence_length = 1024

dataloader = get_sci_bench_dataloader_for_student(tokenizer=tokenizer, source=source, batch_size=batch_size, sequence_length=sequence_length)
print("Dataloader length: ", len(dataloader))

file_path_to_write = "student_generated_cot_text_chemmc.csv"

# Write header and keep file open for the loop
with open(file_path_to_write, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    # Write header
    writer.writerow(['question_id', 'generated_text'])
    
    # Begin evaluation loop
    for batch in tqdm(dataloader, desc="Generating responses"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        problem_ids = batch["problem_id"].to(device)

        # Generate outputs
        # generated_ids = model.generate(
        #     input_ids=batch["input_ids"].squeeze().to("cuda"),
        #     attention_mask=batch["attention_mask"].squeeze().to("cuda"),
        #     pad_token_id=tokenizer.pad_token_id,
        #     max_new_tokens=512,
        # )
        generated_ids = model.generate(
            input_ids=batch["input_ids"].squeeze().to("cuda"),
            attention_mask=batch["attention_mask"].squeeze().to("cuda"),
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=1024,
            temperature=0.7,         # Controls randomness (0.0-1.0). Lower = more focused
            top_p=0.9,              # Nucleus sampling threshold. Higher = more diverse
            top_k=50,               # Limits vocabulary to top K tokens
            do_sample=True,         # Enable sampling (vs greedy decoding)
            repetition_penalty=1.1   # Penalize repeated tokens. >1.0 reduces repetition
        )

        for i in range(batch_size):
            try:
                question_id = tokenizer.decode(problem_ids[i].squeeze(), skip_special_tokens=True)
                generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                input_prompt = tokenizer.decode(input_ids[i].squeeze(), skip_special_tokens=True)
                
                # Clean the text
                question_id = question_id.replace('\n', ' ').replace('\r', ' ').strip()
                generated_text = generated_text.replace('\n', ' ').replace('\r', ' ').strip()
                
                # Write row
                writer.writerow([question_id, generated_text])
            except IndexError:
                # Handle case where last batch might be smaller
                print(f"Skipping incomplete batch item {i}")
                continue
            except Exception as e:
                print(f"Error processing batch item {i}: {str(e)}")
                continue

# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model, tokenizer = get_model_and_tokenizer(model_name)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.eval().to(device)
# source = "atkins"
# batch_size = 4
# sequence_length = 1024

# dataloader = get_sci_bench_dataloader_for_student(tokenizer=tokenizer, source=source, batch_size=batch_size, sequence_length=sequence_length)
# print("Dataloader length: ", len(dataloader))

# file_path_to_write = "student_generated_text.csv"
# with open(file_path_to_write, "w") as f:
#     f.write("question_id,generated_text\n")

# # Begin evaluation loop
# for batch in dataloader:
#     input_ids = batch["input_ids"].to(device)
#     attention_mask = batch["attention_mask"].to(device)
#     labels = batch["labels"].to(device)

#     # Generate outputs
#     generated_ids = model.generate(
#         input_ids=batch["input_ids"].squeeze().to("cuda"),
#         attention_mask=batch["attention_mask"].squeeze().to("cuda"),
#         pad_token_id=tokenizer.pad_token_id,
#         max_new_tokens=512,           # Reduced from 2056 - usually sufficient for math problems
#         # num_return_sequences=1,       # Good for single answer generation
#         # no_repeat_ngram_size=3,      # Increased slightly to prevent repetition
#         # temperature=0.1,             # Reduced for more focused/deterministic outputs
#         # top_p=0.9,                  # Added for nucleus sampling
#         # do_sample=True,              # Enable sampling
#         # repetition_penalty=1.2,      # Added to prevent repetition
#         # early_stopping=True,         # Stop when EOS token is generated
#     )

#     # import pdb; pdb.set_trace()
#     # first_question = tokenizer.decode(input_ids[0].squeeze(), skip_special_tokens=True)
#     # generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

#     # import pdb; pdb.set_trace()

#     # first_question = tokenizer.decode(input_ids[1].squeeze(), skip_special_tokens=True)
#     # generated_text = tokenizer.decode(generated_ids[1], skip_special_tokens=True)

#     # import pdb; pdb.set_trace()

#     # We want to write out the generated text to a file
#     # The stucture of the file will be a .csv
#     # The first column will be the question ID
#     # The second column will be the generated text

#     for i in range(batch_size):
#         question_id = tokenizer.decode(input_ids[i].squeeze(), skip_special_tokens=True)
#         generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
#         with open(file_path_to_write, "a") as f:
#             f.write(f"{question_id},{generated_text}\n")


