"""
Inference Script for the CNN GRU and MHA based Fault Detection Model

This script demonstrates how to use the trained model from the initial prototype
to perform fault detection on a complete, raw data sample. It simulates a
real-world protection scenario where the goal is to detect a fault as quickly
as possible within a continuous stream of data.

The workflow is as follows:
  1. Load the saved Keras model and preprocessing components (scalers).
  2. Load a raw Excel data file containing one or more fault events.
  3. Extract a single, complete event (sample) from the raw data.
  4. Process this sample by creating sliding windows and applying the loaded scalers.
  5. Feed the windows sequentially into the model to find the exact point of detection.
  6. Calculate the protection-relevant detection delay.
  7. Visualize the results on the signal waveform.
"""

# --- Core Libraries ---
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os
import matplotlib.pyplot as plt

# --- 1. SETUP ---

# Set random seeds for reproducibility of any random operations.
np.random.seed(42)
tf.random.set_seed(42)


# --- 2. DATA HANDLING FUNCTIONS ---


def extract_samples_from_raw_data(raw_data: pd.DataFrame) -> list:
    """
    Extracts individual, complete event samples from the raw DataFrame.

    An "event" or "sample" corresponds to one unique 'Iteration' in the data,
    representing a full recording cycle.

    Args:
        raw_data (pd.DataFrame): The raw data loaded from an Excel file.

    Returns:
        A list of dictionaries, where each dictionary represents one sample.
    """
    samples = []
    grouped = raw_data.groupby("Iteration")
    for _, group in grouped:
        sample = group.set_index("Type")
        time_arr = sample.loc["time"].iloc[2:-5].to_numpy(dtype=float)
        voltage_arr = sample.loc["voltage"].iloc[2:-5].to_numpy(dtype=float)
        current_arr = sample.loc["current"].iloc[2:-5].to_numpy(dtype=float)
        label_arr = sample.loc["label"].iloc[2:-5].to_numpy(dtype=int)
        samples.append(
            {
                "time": time_arr,
                "voltage": voltage_arr,
                "current": current_arr,
                "label": label_arr,
            }
        )
    return samples


def window_and_scale_sample(sample: dict, scalers: list, window_config: dict) -> tuple:
    """
    Applies the sliding window mechanism and scales the data for model input.

    It is critical that the *same* scalers from the training phase are used here
    to ensure the data distribution matches what the model expects.

    Args:
        sample (dict): A single sample dictionary.
        scalers (list): A list of two fitted scikit-learn StandardScaler objects.
        window_config (dict): A dictionary containing the `window_size`.

    Returns:
        A tuple of (scaled_windows, window_start_indices).
    """
    window_size = window_config["window_size"]
    stride = 5  # Using a fixed stride for inference demonstration.
    signal = np.column_stack((sample["voltage"], sample["current"]))

    # Generate sliding windows from the signal.
    windows, window_start_indices = [], []
    for start in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[start : start + window_size])
        window_start_indices.append(start)

    # Scale each window using the loaded scalers.
    scaled_windows = []
    for window in windows:
        # Reshape is required because scalers expect 2D array.
        voltage_scaled = scalers[0].transform(window[:, 0].reshape(-1, 1)).flatten()
        current_scaled = scalers[1].transform(window[:, 1].reshape(-1, 1)).flatten()
        scaled_windows.append(np.column_stack((voltage_scaled, current_scaled)))

    return np.array(scaled_windows), window_start_indices


# --- 3. CORE INFERENCE LOGIC ---


def detect_faults_in_sample(
    model: tf.keras.Model, sample: dict, scalers: list, window_config: dict
) -> dict:
    """
    Processes a complete sample window-by-window to find the first fault detection.

    This function simulates a protection relay by analyzing the signal sequentially
    and calculating the delay from the actual fault inception to the model's
    detection.

    Args:
        model: The trained Keras model.
        sample: A dictionary containing the full signal data.
        scalers: The list of loaded scalers.
        window_config: The dictionary with windowing parameters.

    Returns:
        A dictionary containing detailed results of the detection process.
    """
    # First, prepare all windows from the sample.
    scaled_windows, window_start_indices = window_and_scale_sample(
        sample, scalers, window_config
    )
    window_size = window_config["window_size"]

    fault_detected = False
    fault_time, fault_type, detection_delay_ms = None, None, None

    # Process each window sequentially, as a real-time system would.
    for i, window in enumerate(scaled_windows):
        # Make a prediction on the current window.
        pred = model.predict(np.expand_dims(window, axis=0), verbose=0)

        # The output format can vary; handle list (dual-output) or single array.
        pred_output = pred[1] if isinstance(pred, list) else pred
        pred_class = np.argmax(pred_output[0])
        confidence = float(np.max(pred_output[0])) * 100

        # Check if this is the FIRST time a fault has been detected.
        if not fault_detected and pred_class != 0:
            fault_detected = True
            fault_type = pred_class

            # For protection, the detection time is the time at the END of the window
            # that triggered the alarm.
            window_end_idx = min(
                window_start_indices[i] + window_size - 1, len(sample["time"]) - 1
            )
            fault_time = sample["time"][window_end_idx]

            # Find the ground truth fault time from the data labels.
            actual_fault_indices = np.where(sample["label"] > 0)[0]
            if len(actual_fault_indices) > 0:
                actual_fault_time = sample["time"][actual_fault_indices[0]]
                # The total delay is the time from fault inception to the end of the
                # processing window.
                detection_delay_ms = (fault_time - actual_fault_time) * 1000

            # Print a detailed alert for the first detection.
            class_names = {0: "No Fault", 1: "PP Fault", 2: "PPG Fault", 3: "NPG Fault"}
            print(f"\nFAULT DETECTED in window #{i + 1}:")
            print(f"  Type: {class_names.get(fault_type, 'Unknown')}")
            print(f"  Detection Time (Window End): {fault_time:.4f} seconds")
            print(f"  Confidence: {confidence:.2f}%")
            if detection_delay_ms is not None:
                print(f"  Protection-Relevant Delay: {detection_delay_ms:.2f} ms")

            # Since we only care about the first detection, we can stop here.
            break

    # Summarize the final outcome after processing the entire sample.
    print("\n===== DETECTION SUMMARY =====")
    actual_fault_indices = np.where(sample["label"] > 0)[0]
    if len(actual_fault_indices) > 0:
        actual_fault_time = sample["time"][actual_fault_indices[0]]
        if fault_detected:
            print("SUCCESS: Fault detected by model.")
            print(
                f"  Actual Inception: {actual_fault_time:.5f}s | Model Detection: {fault_time:.5f}s"
            )
        else:
            print(
                f"FAILURE: Model FAILED to detect fault that occurred at {actual_fault_time:.4f}s."
            )
    else:
        print("INFO: No fault was present in this sample's labels.")
        if fault_detected:
            print("WARNING: Model registered a FALSE POSITIVE detection.")

    return {
        "fault_detected": fault_detected,
        "fault_time": fault_time,
        "detection_delay_ms": detection_delay_ms,
    }


# --- 4. MAIN EXECUTION BLOCK ---


def main():
    """
    Main function to orchestrate the loading of artifacts and execution of
    the inference pipeline on a sample data file.
    """
    # Define paths to the required artifacts.
    # Note: This script assumes a different artifact structure than the final version.
    preprocessing_dir = "preprocessing_components"
    model_path = "best_amrwavenet_model.h5"  # This V1 prototype saved .h5 files

    # 1. Load all necessary components (model, scalers, config).
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

        scalers = [
            joblib.load(os.path.join(preprocessing_dir, "scaler_0.save")),
            joblib.load(os.path.join(preprocessing_dir, "scaler_1.save")),
        ]
        print("Scalers loaded successfully.")

        with open(os.path.join(preprocessing_dir, "window_config.json"), "r") as f:
            window_config = json.load(f)
        print("Window configuration loaded.")
    except Exception as e:
        print(f"Error loading components: {e}. Ensure training artifacts exist.")
        return

    # 2. Load the raw data to be analyzed.
    try:
        # We will use 'ppg.xlsx' as the example input file.
        test_data_path = "data/ppg.xlsx"
        test_data = pd.read_excel(test_data_path)
        print(f"Test data loaded from '{test_data_path}'.")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # 3. Extract all event samples from the file.
    samples = extract_samples_from_raw_data(test_data)
    print(f"Extracted {len(samples)} complete samples from the data file.")

    # 4. Select a sample to process (e.g., the first one).
    if not samples:
        print("No samples found in the data file.")
        return
    sample_to_test = samples[0]

    # 5. Run the detection logic on the selected sample.
    results = detect_faults_in_sample(model, sample_to_test, scalers, window_config)

    # 6. Visualize the results.
    plt.figure(figsize=(15, 8))
    actual_fault_indices = np.where(sample_to_test["label"] > 0)[0]
    actual_fault_time = (
        sample_to_test["time"][actual_fault_indices[0]]
        if len(actual_fault_indices) > 0
        else None
    )

    # Plot Voltage Signal
    plt.subplot(2, 1, 1)
    plt.plot(sample_to_test["time"], sample_to_test["voltage"], label="Voltage")
    if actual_fault_time:
        plt.axvline(
            x=actual_fault_time,
            color="g",
            linestyle="--",
            label=f"Actual Fault ({actual_fault_time:.4f}s)",
        )
    if results["fault_detected"]:
        plt.axvline(
            x=results["fault_time"],
            color="r",
            linestyle=":",
            label=f"Detected Fault ({results['fault_time']:.4f}s)",
        )
    plt.title("Voltage Signal and Fault Detection", fontsize=16)
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid(True)

    # Plot Current Signal
    plt.subplot(2, 1, 2)
    plt.plot(
        sample_to_test["time"],
        sample_to_test["current"],
        label="Current",
        color="orange",
    )
    if actual_fault_time:
        plt.axvline(
            x=actual_fault_time,
            color="g",
            linestyle="--",
            label=f"Actual Fault ({actual_fault_time:.4f}s)",
        )
    if results["fault_detected"]:
        plt.axvline(
            x=results["fault_time"],
            color="r",
            linestyle=":",
            label=f"Detected Fault ({results['fault_time']:.4f}s)",
        )
    plt.title("Current Signal and Fault Detection", fontsize=16)
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("v1_inference_results.png")
    print("\nVisualization saved to 'v1_inference_results.png'")


if __name__ == "__main__":
    main()
