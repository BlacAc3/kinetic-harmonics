import pandas as pd
import numpy as np
import os
import io

def create_time_series_csv(input_file, output_file, time_interval=0.03):
    """
    Reads raw movement data from a CSV file, adds a time series column,
    and saves the result to a new CSV file.

    Args:
        input_file (str): The name of the input CSV file.
        output_file (str): The name of the output CSV file.
        time_interval (float): The time interval between data points.
    """
    try:
        # Read the data from the input CSV file into a pandas DataFrame
        df = pd.read_csv(input_file)

        # Determine the number of data points
        num_points = len(df)

        # Generate a time column with the specified interval
        time_column = np.arange(0, num_points * time_interval, time_interval)

        # Ensure the time column has the same length as the DataFrame
        if len(time_column) > num_points:
            time_column = time_column[:num_points]

        # Insert the 'time' column as the first column of the DataFrame
        df.insert(0, 'time', time_column)

        # Save the new DataFrame to a CSV file
        df.to_csv(output_file, index=False)

        print(f"Data successfully converted and saved to '{output_file}'")
        print("Generated Time Series Data:")
        print(df)

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Your input data as a multi-line string

    # Define file names
    input_file = 'walk3_angles_corrected.csv'
    output_file = 'optimum_data_time_series.csv'


    # Call the function with file paths
    create_time_series_csv(input_file, output_file)
