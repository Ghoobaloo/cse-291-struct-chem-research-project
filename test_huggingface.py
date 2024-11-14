import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    GenerationConfig,
)

from datasets import load_dataset

model_name = "google/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

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
):
    def tokenize(sample):
        result = tokenizer.__call__(
            sample["problem_text"],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_problem_id = tokenizer.__call__(
            sample["problemid"],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result["labels"] = tokenized_problem_id["input_ids"].clone()
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
):
    dataset, data_collator = get_sci_bench_datasets_for_student(tokenizer, source)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    return dataloader

###########################################
# Prompt model with batches of the problems

# Get the dataloader for the student
dataloader = get_sci_bench_dataloader_for_student(tokenizer, source="atkins")

print("Dataloader length: ", len(dataloader))

# create iterator for the dataloader
dataloader = iter(dataloader)
device = "cude" if torch.cuda.is_available() else "cpu"
model.eval().to(device)
#model.eval().to("cuda")

# iterate over the dataloader
for i in range(2):
    batch = next(dataloader)
    print("Batch: ", batch)
    print("Batch keys: ", batch.keys())
    print("Batch input_ids: ", batch["input_ids"])
    print("Batch attention_mask: ", batch["attention_mask"])
    print("Batch labels: ", batch["labels"])
    # pass in the input_ids and attention_mask to the model
    # print the dimensions of the batch 
    print("Batch input_ids shape: ", batch["input_ids"].shape)
    print("Batch attention_mask shape: ", batch["attention_mask"].shape)
    print("Batch labels shape: ", batch["labels"].shape)

    # the shape is currently 8, 1, 512
    # we need to reshape the input_ids and attention_mask to 8, 512

    # generated_ids = model.generate(
    #     input_ids=batch["input_ids"].squeeze().to("cuda"),
    #     # attention_mask=batch["attention_mask"].squeeze().to("cuda"),
    #     # max_length=512,  # adjust as needed
    #     # num_return_sequences=1,
    #     # no_repeat_ngram_size=2,
    #     # temperature=0.7,
    # )
    print('started generation')
    output = tokenizer.decode(
        model.generate(
            input_ids=batch["input_ids"].squeeze().to(device),
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=1024,
            # logits_processor=[
            #     FlippedLogitsProcessor()
            # ],
            generation_config=GenerationConfig(do_sample=False),
        )[-1],
        skip_special_tokens=True,
    )
    print('done generation')
    #generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # decode the model outputs (this is inference)
    import pdb; pdb.set_trace()