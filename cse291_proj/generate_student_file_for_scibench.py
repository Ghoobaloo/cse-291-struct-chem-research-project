import torch
import csv
from tqdm import tqdm  # Added for progress tracking

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

from student_dataloader import get_sci_bench_dataloader_for_student
from baseline_methods import baseline_cot, baseline_pot
from struct_chem import struct_chem
from struct_chem_optimized import struct_chem_beam

METHOD_TABLE = {
    "CoT": baseline_cot,
    "PoT": baseline_pot,
    "StructChem": struct_chem,
    "StructChemV2": struct_chem_beam,
}

def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def main():
    #model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    model, tokenizer = get_model_and_tokenizer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    source = "chemmc"     #"chemmc" #, "quan", "matter", "atkins"
    batch_size = 1 # Keep batch size fixed at 1 for now
    sequence_length = 512
    method_name = "CoT" #"PoT", "StructChem", "StructChemV2"
    method = METHOD_TABLE[method_name]

    dataloader = get_sci_bench_dataloader_for_student(tokenizer=tokenizer, source=source, method=method_name, batch_size=batch_size, sequence_length=sequence_length)
    print("Dataloader length: ", len(dataloader))

    file_path_to_write = method_name + source + "small" + ".csv"

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
            # # generated_ids = model.generate(
            # #     input_ids=batch["input_ids"].squeeze().to("cuda"),
            # #     attention_mask=batch["attention_mask"].squeeze().to("cuda"),
            # #     pad_token_id=tokenizer.pad_token_id,
            # #     max_new_tokens=512,
            # # )
            # generated_ids = model.generate(
            #     input_ids=batch["input_ids"].squeeze().to("cuda"),
            #     attention_mask=batch["attention_mask"].squeeze().to("cuda"),
            #     pad_token_id=tokenizer.pad_token_id,
            #     max_new_tokens=1024,
            #     temperature=0.7,         # Controls randomness (0.0-1.0). Lower = more focused
            #     top_p=0.9,              # Nucleus sampling threshold. Higher = more diverse
            #     top_k=50,               # Limits vocabulary to top K tokens
            #     do_sample=True,         # Enable sampling (vs greedy decoding)
            #     repetition_penalty=1.1   # Penalize repeated tokens. >1.0 reduces repetition
            # )
            generated_text = method(model, tokenizer, device, input_ids, attention_mask)

            for i in range(batch_size):
                try:
                    question_id = tokenizer.decode(problem_ids[i].squeeze(), skip_special_tokens=True)
                    # generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                    # input_prompt = tokenizer.decode(input_ids[i].squeeze(), skip_special_tokens=True)
                    
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

if __name__ == "__main__":
    main()

