# Human Movement Smoothing with Hopf Oscillator

This project processes human movement data, applies a Hopf oscillator to smooth out the data, and generates visualizations of the data before and after smoothing. The goal is to eliminate jerkiness in the movement data, making it suitable for machine replication.

---

## Project Structure

The project is organized as follows:

```
oscillator v2/
│
├── walk3_angles_corrected.csv
│   The input CSV file containing human movement data.
│
├── images/
│   Directory to store the generated visualizations.
│   - `before.png`: Visualization of the raw input data.
│   - `after.png`: Visualization of the smoothed data.
│
├── src/
│   Source code for the project.
│   ├── hopf_oscillator/
│   │   Contains the implementation of the Hopf oscillator.
│   │   - `hopf.py`: Core logic for the Hopf oscillator.
│   │
│   ├── utils/
│   │   Utility functions for data processing and visualization.
│   │   - `data_processor.py`: Prepares the data with a time interval of 0.03333s.
│   │   - `visualizer.py`: Generates visualizations of the data.
│   │
│   └── main.py
│       Entry point for the project. Orchestrates data processing, smoothing, and visualization.
│
└── README.md
    This file. Provides an overview of the project.
```

---

## How It Works

1. **Input Data**:
   - The project starts with the `walk3_angles_corrected.csv` file, which contains human movement data.

2. **Data Preparation**:
   - The `data_processor.py` script ensures the data has a consistent time interval of 0.03333 seconds between each value.

3. **Hopf Oscillator**:
   - The `hopf.py` script applies the Hopf oscillator to smooth the movement data.

4. **Visualization**:
   - The `visualizer.py` script generates two images:
     - `before.png`: Raw input data.
     - `after.png`: Smoothed data after applying the Hopf oscillator.

5. **Extensibility**:
   - The project is designed to allow additional oscillators to be added for comparison. Simply create a new module in the `hopf_oscillator` directory and integrate it into `main.py`.

---

## How to Run

1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project:
   ```bash
   python src/main.py
   ```
4. Check the `images/` directory for the generated visualizations.

---

## Adding New Oscillators

To add a new oscillator:
1. Create a new Python file in the `src/hopf_oscillator/` directory.
2. Implement the oscillator logic in the new file.
3. Update `main.py` to include the new oscillator for comparison.

---

## Dependencies

The project uses the following Python libraries:
- `numpy`: For numerical computations.
- `matplotlib`: For generating visualizations.
- `pandas`: For data manipulation.

Install these dependencies using the provided `requirements.txt` file.

---

## Future Work

- Add more oscillators for comparison.
- Enhance the visualizations with interactive plots.
- Optimize the Hopf oscillator for real-time applications.

---

## Author

This project was developed by an expert engineer to demonstrate the application of Hopf oscillators in smoothing human movement data.