import torch
import csv
import pandas as pd
import re


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    GenerationConfig,
)

from datasets import load_dataset

# model_name = "google/gemma-2-2b-it"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

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

# Function to extract the last number from the text
def extract_last_number(text):
    # Regex to find all numbers in the text (including floats and negatives)
    matches = re.findall(r"-?\d+\.?\d*", text)
    if matches:
        try:
            return float(matches[-1])  # Return the last number as a float
        except ValueError:
            return None
    return None

# Apply the extraction and calculate correctness
def process_results(df):
    # Debug: Check a sample of generated_text
    print("Sample generated_text:\n", df["generated_text"].iloc[0])

    # Extract the last number
    df["boxed_answer"] = df["generated_text"].apply(extract_last_number)
    
    # Debug: Check if extraction is working
    print("Sample extracted answers:\n", df[["problemid", "boxed_answer"]].head())

    # Compare answer_number with boxed_answer
    df["correct"] = df.apply(lambda row: row["answer_number"] == row["boxed_answer"], axis=1)
    
    # Save the DataFrame to results.csv with additional columns
    results = df[["problemid", "answer_number", "boxed_answer", "correct"]]
    results.to_csv("results_pot.csv", index=False)
    print("Results saved to 'results.csv'")

dataset = load_sci_bench_dataset(source="atkins")
print(dataset)
# Call the function to save the train split of the dataset
save_dataset_to_csv(dataset, "atkins.csv")

# Merge student answer and correct answer
# Load the dataset.csv file
dataset_df = pd.read_csv("atkins.csv")

# Load the student.csv file
# Assumes outputted file from student is called student.csv
student_df = pd.read_csv("./cse291_proj/student_generated_pot_text_atkins.csv")
print(student_df)

# Test for number of matches
# Standardize columns by stripping spaces and converting to lowercase
dataset_df["problemid"] = dataset_df["problemid"].str.strip().str.lower()
student_df["question_id"] = student_df["question_id"].str.strip().str.lower()

# Check for exact matches
problemids = set(dataset_df["problemid"].unique())
question_ids = set(student_df["question_id"].unique())

# Find common values
matches = problemids.intersection(question_ids)
print(f"Number of matches: {len(matches)}")


# Merge the two dataframes on the "problemid" column
merged_df = pd.merge(dataset_df, student_df, left_on="problemid", right_on="question_id")
print(merged_df)

# Process the DataFrame and save results
process_results(merged_df)

print(merged_df["generated_text"].iloc[0])



