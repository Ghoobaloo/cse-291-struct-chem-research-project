import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from prompts.baseline import prompt_config

# Download the SciBench dataset to set up an evaluation pipeline
def load_sci_bench_dataset(source: str = None):
    dataset = load_dataset("xw27/scibench")
    # return portion of the dataset where source is equal to source
    if source:
        dataset = dataset["train"].filter(lambda example: example["source"] == source)
    return dataset

def get_sci_bench_datasets_for_student(
    tokenizer: AutoTokenizer,
    source: str = None,
    sequence_length: int = 512,
):
    def tokenize(sample):
        # Grab the system prompt from the baseline config under the key "prompt_template"
        prompt_template = prompt_config["cot_prompt_template"]

        # Create messages using the LLaMA prompt template
        messages = [
            {"role": "system", "content": prompt_template},
            {"role": "student", "content": sample["problem_text"]},
        ]

        # Concatenate the prompt template with two newlines and the problem text
        formatted_problem = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        result = tokenizer.__call__(
            formatted_problem,
            max_length=sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_problem_id = tokenizer.__call__(
            sample["problemid"],
            max_length=sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result["problem_id"] = tokenized_problem_id["input_ids"].clone()
        return result
    
    dataset = load_sci_bench_dataset(source)
    dataset = dataset.map(tokenize)
    # remove other columns
    dataset = dataset.remove_columns(["solution", "problem_text", "answer_latex", "comment", "answer_number", "unit", "source", "problemid"])
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")
    return dataset, data_collator

def get_sci_bench_dataloader_for_student(
    tokenizer: AutoTokenizer,
    source: str = None,
    batch_size: int = 2,
    sequence_length: int = 512,
):
    dataset, data_collator = get_sci_bench_datasets_for_student(tokenizer, source, sequence_length)

    sampler = torch.utils.data.RandomSampler(dataset, replacement=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=data_collator,
    )
    return dataloader
