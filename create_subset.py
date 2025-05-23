import pandas as pd

# Define file paths
input_file = "/home/ubuntu/verLexperiments/data/train_rloo_subsets.parquet"
output_file = "/home/ubuntu/verLexperiments/data/train_rloo_subsets_15k.parquet"
num_rows = 15000

print(f"Reading {input_file}...")
df = pd.read_parquet(input_file)

print(f"Original dataframe shape: {df.shape}")

# Take the first num_rows
df_subset = df.head(num_rows)

print(f"Subset dataframe shape: {df_subset.shape}")

print(f"Saving subset to {output_file}...")
df_subset.to_parquet(output_file)

print("Done.") 