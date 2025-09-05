import pandas as pd
import numpy as np

def process_data(input_csv, time_interval=0.03333):
    """
    Processes the input data to ensure a consistent time interval between each value.

    Args:
        input_csv (str): Path to the input CSV file.
        time_interval (float): Desired time interval between data points (in seconds).

    Returns:
        pd.DataFrame: Processed DataFrame with interpolated time intervals.
    """
    # Load the data
    data = pd.read_csv(input_csv)

    # Ensure the CSV has a time column or create one
    if 'time' not in data.columns:
        data['time'] = np.arange(0, len(data) * time_interval, time_interval)[:len(data)]

    # Set the time column as the index
    data.set_index('time', inplace=True)

    # Create a new time index with the desired interval
    new_time_index = np.arange(data.index.min(), data.index.max(), time_interval)

    # Reindex the data to the new time index, interpolating missing values
    processed_data = data.reindex(new_time_index).interpolate(method='linear')

    # Reset the index for the processed data
    processed_data.reset_index(inplace=True)
    processed_data.rename(columns={'index': 'time'}, inplace=True)

    return processed_data

if __name__ == "__main__":
    # Example usage
    input_csv_path = "oscillator v2/walk3_angles_corrected.csv"
    output_csv_path = "oscillator v2/processed_data.csv"

    # Process the data
    processed_data = process_data(input_csv_path)

    # Save the processed data to a new CSV
    processed_data.to_csv(output_csv_path, index=False)
    print(f"Processed data saved to {output_csv_path}")
