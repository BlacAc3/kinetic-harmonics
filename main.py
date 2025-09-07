import os
import pandas as pd
from utils.visualizer import plot_data
from hopf_oscillator import HopfOscillator  # Import the new HopfOscillator

def main():
    # Paths
    # input_csv = "optimum_data_time_series.csv"  # Update this if your CSV file is named differently
    # input_csv ="gait_processed.csv"
    input_csv="gait_processed2.csv"
    output_dir = "images"
    before_image_path = os.path.join(output_dir, "before.png")
    after_image_path = os.path.join(output_dir, "after.png")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load the data
    print("Loading data...")
    data = pd.read_csv(input_csv)

    # Step 2: No data preparation needed
    print("No data preparation needed...")
    prepared_data = data

    # Ensure only numeric columns are passed to the HopfOscillator
    # (If your CSV contains non-numeric columns like time or labels, they will be excluded)
    numeric_data = prepared_data.select_dtypes(include='number')
    if numeric_data.shape[1] != prepared_data.shape[1]:
        print("Warning: Non-numeric columns detected and excluded from smoothing.")


    # Step 3: Apply the Hopf Oscillator
    print("Applying Hopf Oscillator...")
    dt = 0.01  # Set the dt value
    hopf = HopfOscillator(dt=dt, coupling_strength=0.1)
    smoothed_data = hopf.run(numeric_data.to_numpy())  # Use the run method

    # Step 4: Visualize the raw data and smoothed data
    print("Generating before and after visualization...")
    plot_data(before_data=numeric_data, after_data=smoothed_data, before_path=before_image_path, after_path=after_image_path, csv_name=input_csv, side_by_side=True)

    print("Processing complete. Check the 'images' directory for results.")

if __name__ == "__main__":
    main()
