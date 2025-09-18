"""
Initial Prototype for a Two-Stage Fault Detection System

This script represents the first version of the fault detection project. It
establishes the core concepts of a two-stage detection mechanism:
  1. A rapid, lightweight model for initial fault detection.
  2. A more complex, multi-branch model (AMRWaveNet) for detailed fault
     classification.

Key Features of this Prototype:
- Multi-resolution analysis (micro, original, macro windows).
- Use of wavelet transforms for feature extraction.
- Custom Keras layers for building blocks like Inception and TCN.
- A dual-output model for simultaneous binary and multiclass classification.

This file serves as a valuable reference for the project, showcasing the
foundational ideas that were later refined and optimized in transformer based model.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import pywt
from sklearn.metrics import accuracy_score
import time

# --- 1. SETUP ---

# Set random seeds for reproducibility of results across runs.
np.random.seed(42)
tf.random.set_seed(42)


# --- 2. DATA PROCESSING MODULE ---


def process_fault_data(
    df: pd.DataFrame,
    label_replacement: dict = None,
    window_size: int = 20,
    stride: int = 5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple:
    """
    Process fault data:
      - Group by 'Iteration', shuffle, and split into train/val/test sets.
      - Extract sliding windows (of size window_size and given stride) from each sample.
      - Optionally replace labels based on label_replacement dict.

    Returns:
      (X_train, y_train, times_train), (X_val, y_val, times_val), (X_test, y_test, times_test)
    """
    # Group data by each simulation run.
    grouped = df.groupby("Iteration")
    iterations = list(grouped.groups.keys())
    np.random.shuffle(iterations)

    # Split the unique iterations into train, validation, and test sets.
    n = len(iterations)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_iter = iterations[:n_train]
    val_iter = iterations[n_train : n_train + n_val]
    test_iter = iterations[n_train + n_val :]

    def extract_windows(iter_keys: list) -> tuple:
        """Helper function to create windows from a list of iteration keys."""
        X_list, y_list, t_list = [], [], []
        for iter_key in iter_keys:
            sample = grouped.get_group(iter_key).set_index("Type")
            # Extract time series data.
            time_row = sample.loc["time"].iloc[2:-5].to_numpy(dtype=float)
            voltage_row = sample.loc["voltage"].iloc[2:-5].to_numpy(dtype=float)
            current_row = sample.loc["current"].iloc[2:-5].to_numpy(dtype=float)
            label_row = sample.loc["label"].iloc[2:-5].to_numpy(dtype=int)

            # Replace labels if needed.
            if label_replacement is not None:
                for old, new in label_replacement.items():
                    label_row[label_row == old] = new

            # Stack voltage and current to form a signal.
            signal = np.stack([voltage_row, current_row], axis=-1)
            num_points = signal.shape[0]

            # Extract sliding windows.
            for start in range(0, num_points - window_size + 1, stride):
                end = start + window_size
                window_input = signal[start:end, :]  # shape: (window_size, 2)
                window_label = label_row[start:end]  # shape: (window_size,)
                t0 = time_row[start]
                X_list.append(window_input)
                y_list.append(window_label)
                t_list.append(t0)
        return np.array(X_list), np.array(y_list), t_list

    X_train, y_train, t_train = extract_windows(train_iter)
    X_val, y_val, t_val = extract_windows(val_iter)
    X_test, y_test, t_test = extract_windows(test_iter)

    return (X_train, y_train, t_train), (X_val, y_val, t_val), (X_test, y_test, t_test)


def generate_multi_resolution_windows(X: np.ndarray, window_size: int) -> tuple:
    """
    Creates three different views (resolutions) from the input windows.

    - Micro: A small, central part of the window for rapid analysis.
    - Original: The window as is.
    - Macro: A padded, larger view to provide more context.

    Args:
        X (np.ndarray): The input windows of shape (n_samples, window_size, n_features).
        window_size (int): The size of the original windows.

    Returns:
        A tuple of (X_micro, X_original, X_macro).
    """
    n_samples = X.shape[0]
    micro_size = max(5, window_size // 2)
    macro_size = min(2 * window_size, window_size + 20)

    # Micro windows (for fast detection) - center part of original window
    start_idx = (window_size - micro_size) // 2
    X_micro = X[:, start_idx : start_idx + micro_size, :]

    # Original windows
    X_original = X

    # Macro windows (for better context) - padded version of original
    X_macro = np.zeros((n_samples, macro_size, X.shape[2]))
    pad_size = (macro_size - window_size) // 2
    X_macro[:, pad_size : pad_size + window_size, :] = X

    return X_micro, X_original, X_macro


def extract_wavelet_features(
    X: np.ndarray, wavelet: str = "db4", level: int = 3
) -> np.ndarray:
    """
    Extracts statistical features using the Discrete Wavelet Transform (DWT).

    Wavelets are effective at decomposing signals into different frequency
    components, which is useful for analyzing electrical fault transients.

    Args:
        X (np.ndarray): The input signal windows.
        wavelet (str): The wavelet family to use (e.g., 'db4').
        level (int): The level of wavelet decomposition.

    Returns:
        A NumPy array of extracted features.
    """
    n_samples, window_size, n_features = X.shape
    wavelet_features = []

    for i in range(n_samples):
        sample_features = []

        for j in range(n_features):  # For both voltage and current
            signal = X[i, :, j]

            # Apply wavelet decomposition
            coeffs = pywt.wavedec(
                signal,
                wavelet,
                level=min(level, pywt.dwt_max_level(len(signal), wavelet)),
            )

            # Extract statistical features from each coefficient set
            for coef in coeffs:
                stats = [
                    np.mean(coef),
                    np.std(coef),
                    np.sum(coef**2),  # Energy
                    -np.sum((coef**2) * np.log(coef**2 + 1e-10)),  # Entropy
                ]
                sample_features.extend(stats)

        wavelet_features.append(sample_features)

    return np.array(wavelet_features)


# --- 3. MODEL ARCHITECTURE BUILDING BLOCKS ---


def create_attention_block(input_tensor):
    """
    Creates a simple self-attention mechanism.

    This block learns to assign importance weights to different time steps,
    allowing the model to focus on the most relevant parts of the signal.
    """
    # Learn a weight for each time step.
    attention_weights = layers.Dense(1, activation="tanh")(input_tensor)
    attention_weights = layers.Flatten()(attention_weights)
    attention_weights = layers.Activation("softmax")(attention_weights)
    # Apply the weights to the input.
    attention_weights = layers.RepeatVector(input_tensor.shape[-1])(attention_weights)
    attention_weights = layers.Permute([2, 1])(attention_weights)
    attention_output = layers.Multiply()([input_tensor, attention_weights])
    # Add a residual connection for stable training.
    output = layers.Add()([input_tensor, attention_output])
    output = layers.LayerNormalization()(output)

    return output


def create_tcn_block(input_tensor, filters, kernel_size, dilation_rate):
    """
    Creates a Temporal Convolutional Network (TCN) block.

    TCNs use dilated convolutions to achieve a large receptive field, enabling
    them to capture long-range dependencies in the time-series data.
    """
    # First dilated causal convolution
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
    )(input_tensor)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Second dilated causal convolution
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
    )(x)

    # Residual connection
    if input_tensor.shape[-1] != filters:
        residual = layers.Conv1D(filters=filters, kernel_size=1)(input_tensor)
    else:
        residual = input_tensor

    x = layers.Add()([x, residual])
    x = layers.LayerNormalization()(x)

    return x


def create_multi_scale_inception_block(input_tensor, filters):
    """
    Creates an Inception-style block for multi-scale feature extraction.

    This block applies multiple convolutions with different kernel sizes in
    parallel, allowing the model to learn features from the signal at
    different time scales simultaneously.
    """
    # 1x1 convolution for dimension reduction
    x_1x1 = layers.Conv1D(filters // 4, kernel_size=1, activation="relu")(input_tensor)

    # Different kernel sizes for multi-scale processing
    x_3x1 = layers.SeparableConv1D(
        filters // 4, kernel_size=3, padding="same", activation="relu"
    )(x_1x1)
    x_5x1 = layers.SeparableConv1D(
        filters // 4, kernel_size=5, padding="same", activation="relu"
    )(x_1x1)
    x_7x1 = layers.SeparableConv1D(
        filters // 4, kernel_size=7, padding="same", activation="relu"
    )(x_1x1)

    # Dilated convolution for enlarged receptive field
    x_3x1_dilated = layers.SeparableConv1D(
        filters // 4, kernel_size=3, dilation_rate=2, padding="same", activation="relu"
    )(x_1x1)

    # Concatenate all branches
    x_concat = layers.Concatenate()([x_3x1, x_5x1, x_7x1, x_3x1_dilated])

    # Add residual connection if shapes match
    if input_tensor.shape[-1] == filters:
        x_out = layers.Add()([input_tensor, x_concat])
    else:
        # If shapes don't match, use 1x1 conv to match dimensions
        x_proj = layers.Conv1D(filters, kernel_size=1)(input_tensor)
        x_out = layers.Add()([x_proj, x_concat])

    x_out = layers.LayerNormalization()(x_out)

    return x_out


# --- 4. MODEL DEFINITIONS ---


def create_rapid_detection_model(input_shape: tuple) -> models.Model:
    """
    Creates a lightweight CNN model for fast binary (fault/no-fault) detection.

    This model is designed for speed and is the first stage in the two-stage
    detection process.

    Args:
        input_shape (tuple): The shape of the input data (e.g., micro-windows).

    Returns:
        A compiled Keras model.
    """
    inputs = layers.Input(shape=input_shape)

    # Simple CNN for fast processing
    x = layers.Conv1D(32, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Binary output (fault/no-fault)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_amrwavenet_model(
    micro_shape: tuple,
    original_shape: tuple,
    macro_shape: tuple,
    wavelet_shape: int,
    num_classes: int = 4,
) -> models.Model:
    """
    Creates the main AMRWaveNet (Adaptive Multi-Resolution Wavelet Network).

    This is a complex, multi-branch model that fuses information from different
    resolutions and feature types to perform detailed fault classification.

    Args:
        micro_shape (tuple): Input shape for the micro-resolution branch.
        original_shape (tuple): Input shape for the original-resolution branch.
        macro_shape (tuple): Input shape for the macro-resolution branch.
        wavelet_shape (int): Input shape for the wavelet features branch.
        num_classes (int): The number of output classes for classification.

    Returns:
        A compiled, multi-output Keras model.
    """
    # Input layers for different resolutions
    micro_input = layers.Input(shape=micro_shape, name="micro_input")
    original_input = layers.Input(shape=original_shape, name="original_input")
    macro_input = layers.Input(shape=macro_shape, name="macro_input")
    wavelet_input = layers.Input(shape=(wavelet_shape,), name="wavelet_input")

    # === Micro resolution branch ===
    # Quick detection with simpler processing
    micro_x = layers.Conv1D(32, kernel_size=3, activation="relu")(micro_input)
    micro_x = layers.MaxPooling1D(pool_size=2)(micro_x)
    micro_x = layers.Conv1D(64, kernel_size=3, activation="relu")(micro_x)
    micro_x = layers.GlobalAveragePooling1D()(micro_x)
    micro_x = layers.Dense(32, activation="relu")(micro_x)

    # === Original resolution branch ===
    # Multi-scale temporal processing
    original_x = layers.Conv1D(64, kernel_size=3, activation="relu")(original_input)

    # Apply multiple inception blocks
    for i in range(2):
        original_x = create_multi_scale_inception_block(original_x, filters=64)

    # Apply attention mechanism
    original_x = create_attention_block(original_x)
    original_x = layers.GlobalAveragePooling1D()(original_x)
    original_x = layers.Dense(64, activation="relu")(original_x)

    # === Macro resolution branch ===
    # TCN blocks with increasing dilation for wider receptive field
    macro_x = layers.Conv1D(64, kernel_size=3, activation="relu")(macro_input)

    # Apply TCN blocks with increasing dilation
    dilation_rates = [1, 2, 4, 8]
    for dilation_rate in dilation_rates:
        macro_x = create_tcn_block(
            macro_x, filters=64, kernel_size=3, dilation_rate=dilation_rate
        )

    macro_x = create_attention_block(macro_x)
    macro_x = layers.GlobalAveragePooling1D()(macro_x)
    macro_x = layers.Dense(64, activation="relu")(macro_x)

    # === Wavelet features branch ===
    wavelet_x = layers.Dense(128, activation="relu")(wavelet_input)
    wavelet_x = layers.BatchNormalization()(wavelet_x)
    wavelet_x = layers.Dropout(0.3)(wavelet_x)
    wavelet_x = layers.Dense(64, activation="relu")(wavelet_x)

    # === Feature fusion ===
    # Concatenate features from all branches
    combined = layers.Concatenate()([micro_x, original_x, macro_x, wavelet_x])

    # Classification head
    x = layers.Dense(128, activation="relu")(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Binary fault detection output
    binary_output = layers.Dense(1, activation="sigmoid", name="binary_output")(x)

    # Multiclass fault classification output
    multiclass_output = layers.Dense(
        num_classes, activation="softmax", name="multiclass_output"
    )(x)

    # Create model with dual outputs
    model = models.Model(
        inputs=[micro_input, original_input, macro_input, wavelet_input],
        outputs=[binary_output, multiclass_output],
    )

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss={
            "binary_output": "binary_crossentropy",
            "multiclass_output": "sparse_categorical_crossentropy",
        },
        metrics={"binary_output": "accuracy", "multiclass_output": "accuracy"},
        loss_weights={"binary_output": 0.3, "multiclass_output": 0.7},
    )

    return model


# --- 5. MAIN TRAINING AND EVALUATION SCRIPT ---


def prepare_data_and_train():
    """
    Orchestrates the entire pipeline from data loading to model training,
    evaluation, and saving.
    """
    # --- 5.1 Data Loading ---
    print("1. Loading data from Excel files...")
    try:
        pp_data = pd.read_excel("data/pp.xlsx")
        ppg_data = pd.read_excel("data/ppg.xlsx")
        npg_data = pd.read_excel("data/npg.xlsx")
        print("   Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 5.2 Data Processing and Splitting ---
    print("2. Processing fault data...")
    (pp_train, pp_val, pp_test) = process_fault_data(
        pp_data, label_replacement=None, window_size=20, stride=5
    )
    (ppg_train, ppg_val, ppg_test) = process_fault_data(
        ppg_data, label_replacement={1: 2}, window_size=20, stride=5
    )
    (npg_train, npg_val, npg_test) = process_fault_data(
        npg_data, label_replacement={1: 3}, window_size=20, stride=5
    )

    # --- 5.3 Dataset Combination ---
    print("3. Combining datasets...")
    X_train = np.concatenate([pp_train[0], ppg_train[0], npg_train[0]], axis=0)
    y_train = np.concatenate([pp_train[1], ppg_train[1], npg_train[1]], axis=0)

    X_val = np.concatenate([pp_val[0], ppg_val[0], npg_val[0]], axis=0)
    y_val = np.concatenate([pp_val[1], ppg_val[1], npg_val[1]], axis=0)

    X_test = np.concatenate([pp_test[0], ppg_test[0], npg_test[0]], axis=0)
    y_test = np.concatenate([pp_test[1], ppg_test[1], npg_test[1]], axis=0)

    print(f"   Training windows: {X_train.shape}, Labels: {y_train.shape}")
    print(f"   Validation windows: {X_val.shape}, Labels: {y_val.shape}")
    print(f"   Test windows: {X_test.shape}, Labels: {y_test.shape}")

    # --- 5.4 Feature Engineering ---
    print("4. Creating multi-resolution windows...")
    # Generate windows at different resolutions
    X_micro_train, X_original_train, X_macro_train = generate_multi_resolution_windows(
        X_train, X_train.shape[1]
    )
    X_micro_val, X_original_val, X_macro_val = generate_multi_resolution_windows(
        X_val, X_val.shape[1]
    )
    X_micro_test, X_original_test, X_macro_test = generate_multi_resolution_windows(
        X_test, X_test.shape[1]
    )

    print("5. Converting window labels to single labels...")
    # Convert window labels to single labels (most frequent class in window)
    y_train_single = np.array(
        [np.bincount(y_train[i]).argmax() for i in range(len(y_train))]
    )
    y_val_single = np.array([np.bincount(y_val[i]).argmax() for i in range(len(y_val))])
    y_test_single = np.array(
        [np.bincount(y_test[i]).argmax() for i in range(len(y_test))]
    )

    # Create binary labels (0=no fault, 1=fault)
    y_train_binary = (y_train_single > 0).astype(int)
    y_val_binary = (y_val_single > 0).astype(int)
    y_test_binary = (y_test_single > 0).astype(int)

    print("6. Extracting wavelet features...")
    # Extract wavelet features
    wavelet_features_train = extract_wavelet_features(X_train)
    wavelet_features_val = extract_wavelet_features(X_val)
    wavelet_features_test = extract_wavelet_features(X_test)

    print(f"   Wavelet features shape: {wavelet_features_train.shape}")

    # --- 5.5 Rapid Model Training ---
    print("7. Creating and training rapid detection model...")
    rapid_model = create_rapid_detection_model(X_micro_train.shape[1:])

    rapid_history = rapid_model.fit(
        X_micro_train,
        y_train_binary,
        validation_data=(X_micro_val, y_val_binary),
        epochs=20,
        batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True
            )
        ],
        verbose=2,
    )

    # Evaluate rapid detection model
    rapid_results = rapid_model.evaluate(X_micro_test, y_test_binary)
    print(f"   Rapid detection model accuracy: {rapid_results[1]:.4f}")

    # --- 5.6 AMRWaveNet Model Training ---
    print("8. Creating and training AMRWaveNet model...")
    # Create full model
    full_model = create_amrwavenet_model(
        X_micro_train.shape[1:],
        X_original_train.shape[1:],
        X_macro_train.shape[1:],
        wavelet_features_train.shape[1],
        num_classes=4,
    )

    # Train full model
    full_history = full_model.fit(
        {
            "micro_input": X_micro_train,
            "original_input": X_original_train,
            "macro_input": X_macro_train,
            "wavelet_input": wavelet_features_train,
        },
        {"binary_output": y_train_binary, "multiclass_output": y_train_single},
        validation_data=(
            {
                "micro_input": X_micro_val,
                "original_input": X_original_val,
                "macro_input": X_macro_val,
                "wavelet_input": wavelet_features_val,
            },
            {"binary_output": y_val_binary, "multiclass_output": y_val_single},
        ),
        epochs=50,
        batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_multiclass_output_accuracy",
                patience=10,
                restore_best_weights=True,
                mode="max",
            ),
            callbacks.ModelCheckpoint(
                "best_amrwavenet_model.h5",
                monitor="val_multiclass_output_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ],
        verbose=2,
    )

    # --- 5.7 Evaluation ---
    print("9. Evaluating models...")
    # Evaluate full model
    full_results = full_model.evaluate(
        {
            "micro_input": X_micro_test,
            "original_input": X_original_test,
            "macro_input": X_macro_test,
            "wavelet_input": wavelet_features_test,
        },
        {"binary_output": y_test_binary, "multiclass_output": y_test_single},
    )

    print("\n=== Model Performance ===")
    print(f"Rapid Detection Model - Binary Accuracy: {rapid_results[1]:.4f}")
    print(f"AMRWaveNet - Binary Accuracy: {full_results[3]:.4f}")
    print(f"AMRWaveNet - Multiclass Accuracy: {full_results[4]:.4f}")

    # Measure average detection time
    detection_times = []
    for i in range(100):  # Test on 100 samples
        if i < len(X_micro_test):
            start_time = time.time()
            _ = rapid_model.predict(np.expand_dims(X_micro_test[i], axis=0))
            detection_time = (time.time() - start_time) * 1000  # Convert to ms
            detection_times.append(detection_time)

    avg_detection_time = np.mean(detection_times)
    print(f"Average Detection Time: {avg_detection_time:.2f} ms")

    # Test on fault inception scenarios
    print("\n=== Fault Inception Testing ===")
    # Find windows where label changes from 0 to non-zero
    fault_inception_indices = []
    for i in range(len(y_test)):
        if np.any(y_test[i] == 0) and np.any(y_test[i] > 0):
            # This window contains a transition from no-fault to fault
            fault_inception_indices.append(i)

    if fault_inception_indices:
        print(f"Found {len(fault_inception_indices)} fault inception windows")

        # Test rapid detection model on these windows
        X_inception_micro = X_micro_test[fault_inception_indices]
        y_inception_binary = y_test_binary[fault_inception_indices]

        # Predict with rapid model
        inception_preds = rapid_model.predict(X_inception_micro)
        inception_pred_classes = (inception_preds > 0.5).astype(int)

        # Calculate accuracy
        inception_accuracy = accuracy_score(y_inception_binary, inception_pred_classes)
        print(f"Fault Inception Detection Accuracy: {inception_accuracy:.4f}")
    else:
        print("No fault inception windows found in test set")

    print("\n10. Saving models...")
    # Save models
    rapid_model.save("rapid_detection_model.h5")
    full_model.save("amrwavenet_model.h5")

    print("11. Plotting training history...")
    # Plot training history
    plt.figure(figsize=(15, 5))

    # Plot rapid model history
    plt.subplot(1, 3, 1)
    plt.plot(rapid_history.history["accuracy"])
    plt.plot(rapid_history.history["val_accuracy"])
    plt.title("Rapid Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="lower right")

    # Plot full model binary accuracy
    plt.subplot(1, 3, 2)
    plt.plot(full_history.history["binary_output_accuracy"])
    plt.plot(full_history.history["val_binary_output_accuracy"])
    plt.title("AMRWaveNet Binary Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="lower right")

    # Plot full model multiclass accuracy
    plt.subplot(1, 3, 3)
    plt.plot(full_history.history["multiclass_output_accuracy"])
    plt.plot(full_history.history["val_multiclass_output_accuracy"])
    plt.title("AMRWaveNet Multiclass Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="lower right")

    plt.tight_layout()
    plt.savefig("training_history.png")

    print("\nComplete! Models saved and ready for deployment.")

    return (
        rapid_model,
        full_model,
        [X_micro_test, X_original_test, X_macro_test, wavelet_features_test],
        [y_test_binary, y_test_single],
    )


def real_time_fault_detector(rapid_model, full_model, test_data=None):
    """
    Simulate real-time fault detection with two-stage approach

    Args:
        rapid_model: The rapid detection model
        full_model: The full AMRWaveNet model
        test_data: Optional test data for simulation

    Returns:
        detection_time: Average detection time in ms
        accuracy: Detection accuracy
    """
    if test_data is None:
        print("No test data provided, exiting")
        return

    X_micro_test, X_original_test, X_macro_test, wavelet_features_test = test_data[0]
    y_test_binary, y_test_multiclass = test_data[1]

    # Initialize metrics
    total_time = 0
    correct_detections = 0
    correct_classifications = 0
    total_samples = min(100, len(X_micro_test))  # Test on up to 100 samples

    print(f"\nSimulating real-time fault detection on {total_samples} samples...")

    for i in range(total_samples):
        # Start timing
        start_time = time.time()

        # Stage 1: Rapid Detection
        rapid_pred = rapid_model.predict(
            np.expand_dims(X_micro_test[i], axis=0), verbose=0
        )[0][0]
        is_fault_detected = rapid_pred > 0.5

        # If fault detected, activate Stage 2
        fault_type = 0  # Default: no fault
        if is_fault_detected:
            # Stage 2: Detailed Classification
            full_pred = full_model.predict(
                {
                    "micro_input": np.expand_dims(X_micro_test[i], axis=0),
                    "original_input": np.expand_dims(X_original_test[i], axis=0),
                    "macro_input": np.expand_dims(X_macro_test[i], axis=0),
                    "wavelet_input": np.expand_dims(wavelet_features_test[i], axis=0),
                },
                verbose=0,
            )
            fault_type = np.argmax(full_pred[1][0])

        # End timing
        end_time = time.time()
        detection_time = (end_time - start_time) * 1000  # Convert to ms
        total_time += detection_time

        # Check accuracy
        if is_fault_detected == bool(y_test_binary[i]):
            correct_detections += 1

        if fault_type == y_test_multiclass[i]:
            correct_classifications += 1

        # Periodic progress update
        if (i + 1) % 20 == 0 or i == total_samples - 1:
            print(f"  Processed {i + 1}/{total_samples} samples")

    # Calculate metrics
    avg_detection_time = total_time / total_samples
    detection_accuracy = correct_detections / total_samples
    classification_accuracy = correct_classifications / total_samples

    print("\n=== Real-time Simulation Results ===")
    print(f"Average Detection Time: {avg_detection_time:.2f} ms")
    print(f"Detection Accuracy: {detection_accuracy:.4f}")
    print(f"Classification Accuracy: {classification_accuracy:.4f}")

    return avg_detection_time, detection_accuracy, classification_accuracy


if __name__ == "__main__":
    # Run the complete pipeline
    rapid_model, full_model, test_data, test_labels = prepare_data_and_train()

    # Simulate real-time fault detection
    real_time_fault_detector(rapid_model, full_model, [test_data, test_labels])
