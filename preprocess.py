import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- SAFE PATH HANDLING ---
script_dir = os.path.dirname(__file__)
input_file = os.path.join(script_dir, "gait_processed.csv")
output_file = os.path.join(script_dir, "Subject 1_1_preprocessed.csv")

# --- LOAD CSV ---
df = pd.read_csv(input_file)
print("✅ File loaded successfully!")
print(df.head())

# --- EXTRACT L-KNEE JOINT (DEG → RAD) ---
lKnee_deg = df["lKnee"].values
lKnee_rad = np.deg2rad(lKnee_deg)

# --- CREATE NORMALIZED TIME VECTOR (0–1) ---
time_norm = np.linspace(0, 1, len(lKnee_rad))

# --- SAVE PROCESSED DATA ---
processed = pd.DataFrame({
    "time_norm": time_norm,
    "lKnee_rad": lKnee_rad
})
processed.to_csv(output_file, index=False)
print(f"✅ Processed file saved as: {output_file}")
