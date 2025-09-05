import pandas as pd

# Pivot to wide format
def make_wide(d):
    wide = d.pivot_table(
        index='Time',
        columns=['Leg','Joint'],
        values='Angle',
        aggfunc='mean'  # average over repetitions if multiple reps
    )
    # Flatten the MultiIndex
    wide.columns = [f"{leg}{joint}" for leg, joint in wide.columns]
    # Rename columns
    colmap = {
        "11": 'lAnkle', "12": 'lKnee', "13": 'lHip',
        "21": 'rAnkle', "22": 'rKnee', "23": 'rHip'
    }
    wide = wide.rename(columns=colmap)
    return wide.reset_index()

def process_gait_data(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Rename columns
    df = df.rename(columns={
        'time': 'Time',
        'leg': 'Leg',
        'joint': 'Joint',
        'angle': 'Angle'
    })

    # Convert Leg and Joint to numeric
    df['Leg'] = pd.to_numeric(df['Leg'])
    df['Joint'] = pd.to_numeric(df['Joint'])

    # Call make_wide
    wide_df = make_wide(df)

    # Set the column order and names
    wide_df = wide_df[['Time', 'lAnkle', 'lKnee', 'lHip', 'rAnkle', 'rKnee', 'rHip']]
    wide_df.columns = ['Time', 'lAnkle', 'lKnee', 'lHip', 'rAnkle', 'rKnee', 'rHip']

    # Save to CSV
    wide_df.to_csv(output_csv, index=False, sep=',')
if __name__ == '__main__':
    process_gait_data('gait.csv', 'gait_processed.csv')
