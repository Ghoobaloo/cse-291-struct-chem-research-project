import torch
import csv
import pandas as pd

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

# Function to save dataset to CSV
def save_dataset_to_csv(dataset, output_file):
    # Define the fields you want to extract
    fields = ["problemid", "answer_number", "problem_text"]

    # Open the CSV file for writing
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        # Write the header row
        writer.writeheader()

        # Iterate over each example in the dataset
        for example in dataset:
            # Write only the selected fields to the CSV
            writer.writerow({field: example[field] for field in fields})


dataset = load_sci_bench_dataset()
print(dataset["train"][0])
# Call the function to save the train split of the dataset
save_dataset_to_csv(dataset["train"], "dataset.csv")

# Merge student answer and correct answer
# Load the dataset.csv file
dataset_df = pd.read_csv("dataset.csv")

# Load the student.csv file
# Assumes outputted file from student is called student.csv
student_df = pd.read_csv("student.csv")

# Merge the two dataframes on the "problemid" column
merged_df = pd.merge(dataset_df, student_df, on="problemid", how="inner")

# Save the merged dataframe to a new CSV file
merged_df.to_csv("merge.csv", index=False)

# File processor: compare student answer to correct answer and save to file
# Load the merge.csv file
merged_df = pd.read_csv("merge.csv")

# Compare answer_number and student_ans row by row
merged_df["is_correct"] = merged_df.apply(
    lambda row: row["answer_number"] == row["student_ans"], axis=1
)

# Save the results to output.csv
merged_df.to_csv("output.csv", index=False)

print("Comparison results have been saved in 'output.csv'")




