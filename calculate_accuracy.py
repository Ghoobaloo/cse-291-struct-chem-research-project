import pandas as pd

def calculate_accuracy(file_path):
    # Load the file into a DataFrame
    df = pd.read_csv(file_path)

    # Count True and False values in the 'correct' column
    true_count = df['correct'].sum()  # Since True is equivalent to 1, this sums up the True values
    total_count = len(df)

    # Calculate accuracy
    accuracy = true_count / total_count if total_count > 0 else 0

    return accuracy

# Example usage
file_path = "results_cot_atkins.csv"  # Replace with your actual file path
file_path_2 = "results_pot_atkins.csv"
file_path_3 = "results_cot_chemmc.csv"
file_path_4 = "results_pot_chemmc.csv"


accuracy = calculate_accuracy(file_path)
accuracy_2 = calculate_accuracy(file_path_2)
accuracy_3 = calculate_accuracy(file_path_3)
accuracy_4 = calculate_accuracy(file_path_4)

print(f"CoT Atkins Accuracy: {accuracy:.2%}")
print(f"PoT Atkins Accuracy: {accuracy_2:.2%}")
print(f"CoT Chemmc Accuracy: {accuracy_3:.2%}")
print(f"PoT Chemmc Accuracy: {accuracy_4:.2%}")

