print("Is this code segment being run?")

import torch
import csv
from tqdm import tqdm  # Added for progress tracking

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

import re

from student_dataloader import get_sci_bench_dataloader_for_student
from generate_student_file_for_scibench import get_model_and_tokenizer
import prompts.baseline

print("Program should be finished.")