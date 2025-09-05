import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_data(before_data: pd.DataFrame, after_data, before_path: str, after_path: str, max_series: int = 10, enforce_same_y_axis: bool = False, side_by_side: bool = False, csv_name=None):
    """
    Generates visualizations for the data before and after applying the Hopf oscillator. Supports side-by-side column-wise plotting.

    Parameters:
        before_data (pd.DataFrame): The raw input data.
        after_data (pd.DataFrame or np.ndarray): The smoothed data after applying the Hopf oscillator.
        before_path (str): Path to save the raw data image.
        after_path (str): Path to save the smoothed data image.
        max_series (int): Maximum number of series to plot for readability.
    Parameters:
        side_by_side (bool): If True, generates side-by-side plots for each column.
        csv_name (str): Name of the CSV file used for analysis (required for side_by_side).
    """
    if side_by_side:
        if not csv_name:
            raise ValueError("csv_name must be provided when side_by_side=True")
        _plot_side_by_side(before_data, after_data, csv_name, max_series)
        return
    columns = before_data.columns[:max_series]
    colors = plt.cm.get_cmap('tab10', len(columns))

    # Determine global y-axis limits if enforce_same_y_axis is True
    if enforce_same_y_axis:
        y_min = min(before_data.min().min(), after_data.min().min() if isinstance(after_data, pd.DataFrame) else after_data.min())
        y_max = max(before_data.max().max(), after_data.max().max() if isinstance(after_data, pd.DataFrame) else after_data.max())
    else:
        y_min, y_max = None, None

    # Plot the raw input data
    plt.figure(figsize=(10, 6))
    for idx, column in enumerate(columns):
        plt.plot(before_data.index, before_data[column], label=column, color=colors(idx))
    if enforce_same_y_axis:
        plt.ylim(y_min, y_max)
    plt.title("Raw Input Data")
    plt.xlabel("Time Steps")
    plt.ylabel("Angle Values")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.savefig(before_path, bbox_inches='tight')
    plt.close()

    # Plot the smoothed data
    plt.figure(figsize=(10, 6))
    if isinstance(after_data, pd.DataFrame):
        after_columns = after_data.columns[:max_series]
        for idx, column in enumerate(after_columns):
            plt.plot(after_data.index, after_data[column], label=column, color=colors(idx))
    elif isinstance(after_data, np.ndarray):
        num_columns = after_data.shape[1]
        for i in range(min(num_columns, max_series)):
            plt.plot(before_data.index, after_data[:, i], label=before_data.columns[i] if i < len(before_data.columns) else f"Column {i+1}", color=colors(i))
    else:
        raise ValueError("after_data must be a DataFrame or NumPy array")
    if enforce_same_y_axis:
        plt.ylim(y_min, y_max)
    plt.title("Smoothed Data (After Hopf Oscillator)")
    plt.xlabel("Time Steps")
    plt.ylabel("Angle Values")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.savefig(after_path, bbox_inches='tight')
    plt.close()


def _plot_side_by_side(before_data: pd.DataFrame, after_data, csv_name: str, max_series: int):
    """
    Generates side-by-side plots for each column, saving them in organized directories.

    Parameters:
        before_data (pd.DataFrame): The raw input data.
        after_data (pd.DataFrame or np.ndarray): The smoothed data after applying the Hopf oscillator.
        csv_name (str): Name of the CSV file used for analysis.
    """
    base_dir = os.path.splitext(csv_name)[0]
    os.makedirs(base_dir, exist_ok=True)

    for column in before_data.columns:
        column_image_path = os.path.join(base_dir, f"{column}.png")

        # Create a single figure with two subplots: before and after
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

        # Plot the raw input data (Before) on the left
        axes[0].plot(before_data.index, before_data[column], label=f"Before: {column}", color="blue")
        axes[0].set_title(f"Raw Input Data - {column}")
        axes[0].set_xlabel("Time Steps")
        axes[0].set_ylabel("Angle Values")
        axes[0].legend()
        axes[0].grid()
        axes[0].set_ylim(before_data[column].min(), before_data[column].max())

        # Plot the smoothed data (After) on the right
        if isinstance(after_data, pd.DataFrame):
            axes[1].plot(after_data.index, after_data[column], label=f"After: {column}", color="green")
            axes[1].set_ylim(after_data[column].min(), after_data[column].max())
        elif isinstance(after_data, np.ndarray):
            col_idx = before_data.columns.get_loc(column)
            axes[1].plot(before_data.index, after_data[:, col_idx], label=f"After: {column}", color="green")
            axes[1].set_ylim(after_data[:, col_idx].min(), after_data[:, col_idx].max())
        else:
            raise ValueError("after_data must be a DataFrame or NumPy array")
        axes[1].set_title(f"Smoothed Data - {column}")
        axes[1].set_xlabel("Time Steps")
        axes[1].legend()
        axes[1].grid()

        # Adjust layout and save the combined image
        plt.tight_layout()
        plt.savefig(column_image_path, bbox_inches='tight')
        plt.close()
