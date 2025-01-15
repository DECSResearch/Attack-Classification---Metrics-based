import pandas as pd
from datetime import datetime

def process_csv_data(csv_path):
    """
    Process CSV file containing Jetson metrics and sample data every 5 seconds.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Processed and sampled data
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp to ensure proper ordering
    df = df.sort_values('timestamp')
    
    # Set timestamp as index for resampling
    df.set_index('timestamp', inplace=True)
    
    # Sample every 5 seconds
    df_sampled = df.resample('5S').first().dropna()
    
    return df_sampled

# Example usage
if __name__ == "__main__":
    # Process Nano07 data
    try:
        # Process the data for Nano07
        result_nano07 = process_csv_data('nano07.csv')
        
        # Display results for Nano07
        print("\nSampled data for Nano07 (every 5 seconds):")
        print(result_nano07)
        
        # Save the sampled data for Nano07
        output_path_nano07 = 'nano07_short.csv'
        result_nano07.to_csv(output_path_nano07)
        print(f"\nSampled data saved to: {output_path_nano07}")
        
    except FileNotFoundError:
        print("Error: Could not find Nano07 file")
    except Exception as e:
        print(f"Error processing Nano07 file: {str(e)}")

    # Process Nano08 data
    try:
        # Process the data for Nano08
        result_nano08 = process_csv_data('nano08.csv')
        
        # Display results for Nano08
        print("\nSampled data for Nano08 (every 5 seconds):")
        print(result_nano08)
        
        # Save the sampled data for Nano08
        output_path_nano08 = 'nano08_short.csv'
        result_nano08.to_csv(output_path_nano08)
        print(f"\nSampled data saved to: {output_path_nano08}")
        
    except FileNotFoundError:
        print("Error: Could not find Nano08 file")
    except Exception as e:
        print(f"Error processing Nano08 file: {str(e)}")




