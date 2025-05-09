import csv
import random

# Define column names
columns = [
    "sample_id", "condition", "KRAS", "RAF1", "MAP2K1", "MAPK1", "MAPK3", "PIK3CA", "AKT1", "MTOR", "PTEN",
    "TP53", "MDM2", "CDKN1A", "BAX", "WNT1", "FZD1", "APC", "CTNNB1", "APOE", "APP", "PSEN1",
    "MAPT", "TNF", "IL1B", "TREM2", "GFAP"
]

# Define base expression profiles for each condition
base_profiles = {
    "cancer":        [8.23, 5.67, 7.89, 6.54, 4.32, 9.12, 8.45, 5.78, 2.34, 1.23, 6.78, 3.45, 5.67, 7.89, 4.56, 2.34, 8.90, 3.45, 2.34, 5.67, 1.23, 7.89, 4.56, 2.34, 3.45],
    "neurodegenerative": [2.34, 3.45, 2.56, 3.12, 4.56, 3.23, 2.78, 4.12, 6.78, 5.23, 2.34, 5.67, 3.12, 2.34, 3.78, 6.12, 2.34, 8.23, 9.12, 7.89, 8.56, 8.12, 7.56, 6.34, 9.45],
    "cardiovascular":    [3.67, 4.78, 5.89, 4.56, 3.45, 6.12, 5.67, 3.23, 4.56, 5.89, 3.67, 5.12, 4.56, 3.67, 5.34, 5.12, 3.67, 3.45, 3.78, 2.56, 2.34, 6.78, 7.23, 3.12, 4.56],
    "autoimmune":        [5.67, 4.23, 3.56, 5.12, 6.34, 4.56, 3.89, 5.67, 5.12, 4.56, 5.67, 3.89, 5.12, 5.67, 3.23, 4.56, 5.67, 4.23, 4.56, 5.12, 3.89, 9.12, 8.67, 7.89, 6.12],
    "metabolic":         [4.56, 3.12, 6.78, 4.23, 3.56, 5.34, 7.12, 4.56, 3.12, 3.67, 4.56, 6.12, 4.23, 4.56, 6.34, 3.12, 4.56, 5.12, 3.23, 4.56, 5.12, 4.23, 5.12, 3.67, 2.90],
    "infectious":        [6.34, 7.12, 5.67, 6.89, 7.56, 5.89, 4.56, 6.34, 3.89, 2.56, 6.34, 4.56, 6.89, 6.34, 4.23, 2.56, 6.34, 3.89, 5.12, 3.56, 4.23, 8.23, 7.56, 9.12, 8.67],
    "normal":            [1.12, 1.23, 1.34, 1.45, 1.56, 1.23, 1.34, 1.45, 7.12, 1.23, 1.34, 1.45, 1.56, 1.23, 1.34, 7.45, 1.56, 1.23, 1.34, 1.45, 1.56, 1.23, 1.34, 1.45, 1.56],
}

# Define sample counts
sample_counts = {
    "cancer": 500,
    "neurodegenerative": 500,
    "cardiovascular": 500,
    "autoimmune": 500,
    "metabolic": 500,
    "infectious": 400,
    "normal": 100
}

def generate_sample(sample_id, condition, base_values, variation):
    return [sample_id, condition] + [
        round(v + random.uniform(-variation, variation), 2)
        for v in base_values
    ]

# Collect all samples
data = [columns]
sample_index = 1

for condition, count in sample_counts.items():
    base_values = base_profiles[condition]
    variation = 0.5 if condition == "cancer" else 0.3 if condition == "neurodegenerative" else 0.4
    for _ in range(count):
        sample_id = f"GSM{sample_index:04d}"
        row = generate_sample(sample_id, condition, base_values, variation)
        data.append(row)
        sample_index += 1

# Write to CSV
with open("./data/Training.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

print("✅ Dataset generated and saved as 'gene_expression_dataset.csv'")