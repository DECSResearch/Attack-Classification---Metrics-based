import pandas as pd

def process_csv(input_file, output_file):
    # Read the CSV file without parsing dates first
    df = pd.read_csv(input_file)
    
    # Remove first 23 rows
    df_filtered = df.iloc[312:]
    
    # Reset index
    df_filtered = df_filtered.reset_index(drop=True)
    
    # Save to CSV, preserving original format
    df_filtered.to_csv(output_file, index=False)

# Example usage
input_file = 'Nano07V2_gt - Copy.csv'
output_file = 'Nano07V2_gt_processed.csv'

try:
    process_csv(input_file, output_file)
    print(f"Successfully processed {input_file}")
    print(f"Output saved to {output_file}")
except Exception as e:
    print(f"An error occurred: {str(e)}")