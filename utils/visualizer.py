import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_data(before_data: pd.DataFrame, after_data, before_path: str, after_path: str, max_series: int = 10, enforce_same_y_axis: bool = False, side_by_side: bool = False, csv_name=None, adaptive=False):
    """
    Generates visualizations for the data before and after applying the Hopf oscillator.

    Parameters:
        before_data (pd.DataFrame): The raw input data.
        after_data (pd.DataFrame or np.ndarray): The smoothed data after applying the Hopf oscillator.
        before_path (str): Path to save the raw data image.
        after_path (str): Path to save the smoothed data image.
        max_series (int): Maximum number of series to plot for readability.
        enforce_same_y_axis (bool): If True, enforces the same y-axis limits for both plots.
        side_by_side (bool): If True, generates side-by-side plots for each column.
        csv_name (str): Name of the CSV file used for analysis (required for side_by-side).
    """

    if adaptive:
        if not csv_name:
            raise ValueError("csv name not found")
        _plot_side_by_side_adaptive_x(before_data, after_data, csv_name, max_series)
        return


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

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Plot raw and smoothed data on the first subplot
    ax1 = axes[0]
    for idx, column in enumerate(columns):
        ax1.plot(before_data.index, before_data[column], label=f"Before: {column}", color=colors(idx), linestyle='-')

        # Plot the 'after_data' on the same subplot
        if isinstance(after_data, pd.DataFrame):
            ax1.plot(after_data.index, after_data[column], label=f"After: {column}", color=colors(idx), linestyle='--')
        elif isinstance(after_data, np.ndarray):
            col_idx = before_data.columns.get_loc(column)
            ax1.plot(before_data.index, after_data[:, col_idx], label=f"After: {column}", color=colors(idx), linestyle='--')
        else:
            raise ValueError("after_data must be a DataFrame or NumPy array")


    if enforce_same_y_axis:
        ax1.set_ylim(y_min, y_max)
    ax1.set_title("Raw Data vs. Smoothed Data")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Angle Values")
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    ax1.grid()

    # Plot amplitude on the second subplot
    ax2 = axes[1]
    for idx, column in enumerate(columns):
        # Calculate amplitude (e.g., using peak-to-peak amplitude)
        before_amplitude = before_data[column].max() - before_data[column].min()
        if isinstance(after_data, pd.DataFrame):
            after_amplitude = after_data[column].max() - after_data[column].min()
        elif isinstance(after_data, np.ndarray):
            after_amplitude = after_data[:, idx].max() - after_data[:, idx].min()
        else:
            raise ValueError("after_data must be a DataFrame or NumPy array")
        ax2.plot([0], [before_amplitude], marker='o', label=f"Before: {column}", color=colors(idx), markersize=5)
        ax2.plot([1], [after_amplitude], marker='x', label=f"After: {column}", color=colors(idx), markersize=5)

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Before', 'After'])
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Amplitude Comparison")
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    ax2.grid()


    plt.tight_layout()
    plt.savefig(before_path, bbox_inches='tight')
    plt.close()


def _plot_side_by_side(before_data: pd.DataFrame, after_data, csv_name: str, max_series: int):
    """
    Generates plots for each column, saving them in organized directories.
    Each plot contains 'before' and 'after' data, and an amplitude comparison.

    Parameters:
        before_data (pd.DataFrame): The raw input data.
        after_data (pd.DataFrame or np.ndarray): The smoothed data after applying the Hopf oscillator.
        csv_name (str): Name of the CSV file used for analysis.
    """
    base_dir = os.path.splitext(csv_name)[0]
    os.makedirs(base_dir, exist_ok=True)

    for column in before_data.columns:
        image_path = os.path.join(base_dir, f"{column}_comparison.png")

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        # Time Series Plot (Before and After on the same plot)
        ax1.plot(before_data.index, before_data[column], label=f"Before: {column}", color="blue")
        if isinstance(after_data, pd.DataFrame):
            ax1.plot(after_data.index, after_data[column], label=f"After: {column}", color="green")
        elif isinstance(after_data, np.ndarray):
            col_idx = before_data.columns.get_loc(column)
            ax1.plot(before_data.index, after_data[:, col_idx], label=f"After: {column}", color="green")
        else:
            raise ValueError("after_data must be a DataFrame or NumPy array")
        ax1.set_title(f"Time Series Data - {column}")
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Angle Values")
        ax1.legend()
        ax1.grid()

        # Amplitude Comparison Plot
        before_amplitude = before_data[column].max() - before_data[column].min()

        if isinstance(after_data, pd.DataFrame):
            after_amplitude = after_data[column].max() - after_data[column].min()
        elif isinstance(after_data, np.ndarray):
            col_idx = before_data.columns.get_loc(column)
            after_amplitude = after_data[:, col_idx].max() - after_data[:, col_idx].min()
        else:
            raise ValueError("after_data must be a DataFrame or NumPy array")

        ax2.bar(['Before', 'After'], [before_amplitude, after_amplitude], color=['blue', 'green'])
        ax2.set_ylabel("Amplitude")
        ax2.set_title(f"Amplitude Comparison - {column}")
        ax2.grid(axis='y')

        plt.tight_layout()
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

def _plot_side_by_side_adaptive_x(before_data: pd.DataFrame, after_data, csv_name: str, max_series: int):
    """
    Generates plots for each column, saving them in organized directories.
    Each plot contains 'before' and 'after' data in separate charts within the same image.
    X-axis is adaptive to the data.

    Parameters:
        before_data (pd.DataFrame): The raw input data.
        after_data (pd.DataFrame or np.ndarray): The smoothed data after applying the Hopf oscillator.
        csv_name (str): Name of the CSV file used for analysis.
    """
    base_dir = os.path.splitext(csv_name)[0]
    os.makedirs(base_dir, exist_ok=True)

    for column in before_data.columns:
        image_path = os.path.join(base_dir, f"{column}_comparison_adaptive_x.png")

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        # Time Series Plot - Before Data
        ax1.plot(before_data.index, before_data[column], color="blue")
        ax1.set_title(f"Before: {column}")
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Angle Values")
        ax1.grid()
        ax1.relim()  # Update axes limits
        ax1.autoscale_view()

        # Time Series Plot - After Data
        if isinstance(after_data, pd.DataFrame):
            ax2.plot(after_data.index, after_data[column], color="green")
        elif isinstance(after_data, np.ndarray):
            col_idx = before_data.columns.get_loc(column)
            ax2.plot(before_data.index, after_data[:, col_idx], color="green")
        else:
            raise ValueError("after_data must be a DataFrame or NumPy array")
        ax2.set_title(f"After: {column}")
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Angle Values")
        ax2.grid()
        ax2.relim()  # Update axes limits
        ax2.autoscale_view()

        plt.tight_layout()
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()
