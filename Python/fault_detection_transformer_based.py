"""
Fault Detection: Training and Evaluation Pipeline

This script serves as the master training file for the transformer-based fault
detection system. It orchestrates the entire ML pipeline:
  1. Loads and preprocesses raw electrical signal data.
  2. Applies robust data splitting and feature engineering (wavelets).
  3. Builds a state-of-the-art Transformer model.
  4. Trains the model using advanced techniques (Focal Loss, class weights,
     and a learning rate scheduler) to maximize performance.
  5. Evaluates the final model and generates explainable AI (XAI) artifacts,
     including SHAP plots and attention heatmaps.
  6. Logs all results and artifacts to MLflow for experiment tracking.

This file produces the following key artifacts required for deployment:
  - `transformer_model`: The trained Keras model.
  - `scalers.pkl`: The fitted StandardScaler objects for data normalization.
"""

import os
import pickle
import logging
from dataclasses import dataclass, field
from typing import Dict

# Core ML & Data Science Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import pywt
import mlflow
import shap

# Scikit-learn for preprocessing and evaluation
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


@dataclass
class Config:
    """Central configuration for the V4 ML pipeline."""

    # Data source parameters
    data_files: Dict[str, str] = field(
        default_factory=lambda: {
            "pp": "data/pp.xlsx",
            "ppg": "data/ppg.xlsx",
            "npg": "data/npg.xlsx",
        }
    )
    label_map: Dict[str, int] = field(
        default_factory=lambda: {"normal": 0, "pp": 1, "ppg": 2, "npg": 3}
    )
    trim_slice: slice = slice(2, -5)

    # Preprocessing parameters
    window_size: int = 40
    stride: int = 10
    val_size: float = 0.2
    test_size: float = 0.2

    # Wavelet feature parameters
    wavelet_family: str = "db4"
    wavelet_level: int = 4

    # Transformer model architecture parameters
    d_model: int = 64
    num_heads: int = 4
    ff_dim: int = 128
    num_transformer_blocks: int = 3
    transformer_dropout: float = 0.15

    # Training Parameters
    epochs: int = 50
    batch_size: int = 64
    initial_learning_rate: float = 0.001
    use_focal_loss: bool = True
    early_stopping_patience: int = 15
    lr_scheduler_patience: int = 5

    # Environment and XAI parameters
    seed: int = 42
    mlflow_experiment_name: str = "Fault_Detection"
    shap_background_samples: int = 100


# =============================================================================
# 2. DATA PROCESSING
# =============================================================================
def set_seeds(seed: int):
    """Sets random seeds for reproducibility across libraries."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_and_window_data(cfg: Config) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Loads data from Excel files, applies labels, and creates sliding windows.

    Args:
        cfg: The configuration object.

    Returns:
        A tuple of (all_windows, all_labels, all_groups) as NumPy arrays.
    """
    all_windows, all_labels, all_groups = [], [], []
    fault_type_map = {"pp": 1, "ppg": 2, "npg": 3}
    for fault_type, file_path in cfg.data_files.items():
        logging.info(f"Processing {file_path}...")
        try:
            df = pd.read_excel(file_path)
        except FileNotFoundError:
            logging.error(f"FATAL: {file_path} not found.")
            raise
        for iter_key, sample in df.groupby("Iteration"):
            voltage = (
                sample.loc[sample["Type"] == "voltage"]
                .iloc[:, cfg.trim_slice]
                .to_numpy()
                .flatten()
            )
            current = (
                sample.loc[sample["Type"] == "current"]
                .iloc[:, cfg.trim_slice]
                .to_numpy()
                .flatten()
            )
            label = (
                sample.loc[sample["Type"] == "label"]
                .iloc[:, cfg.trim_slice]
                .to_numpy()
                .flatten()
            )
            if fault_type in fault_type_map:
                label[label == 1] = fault_type_map[fault_type]
            signal = np.stack([voltage, current], axis=-1)
            for start in range(0, len(signal) - cfg.window_size + 1, cfg.stride):
                all_windows.append(signal[start : start + cfg.window_size, :])
                all_labels.append(label[start : start + cfg.window_size])
                all_groups.append(f"{fault_type}_{iter_key}")
    return np.array(all_windows), np.array(all_labels), np.array(all_groups)


def extract_wavelet_features(X: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """Extracts statistical wavelet features from a batch of windows."""
    n_samples, _, n_features = X.shape
    features = []
    for i in range(n_samples):
        sample_f = []
        for j in range(n_features):
            coeffs = pywt.wavedec(X[i, :, j], wavelet, level=level)
            for coef in coeffs:
                sample_f.extend([np.mean(coef), np.std(coef), np.sum(np.square(coef))])
        features.append(sample_f)
    return np.array(features)


def preprocess_and_split_data(cfg: Config) -> (Dict, Dict, Dict, Dict):
    """
    Main function to orchestrate the entire data preprocessing and splitting pipeline.
    """
    X, y_windowed, groups = load_and_window_data(cfg)

    # Convert windowed labels to a single label per window.
    # We use np.max for high sensitivity: if any fault is present, the window is labeled as a fault.
    y = np.array([np.max(w) for w in y_windowed])
    y_binary = (y > 0).astype(int)

    # Use GroupShuffleSplit to ensure all windows from a single event remain in the same data split.
    # This is CRITICAL to prevent data leakage and get a trustworthy evaluation.
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=cfg.val_size + cfg.test_size, random_state=cfg.seed
    )
    train_idx, temp_idx = next(gss_val.split(X, y, groups))
    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.test_size / (cfg.val_size + cfg.test_size),
        random_state=cfg.seed,
    )
    val_idx, test_idx = next(gss_test.split(X[temp_idx], y[temp_idx], groups[temp_idx]))

    X_train, y_train, y_train_binary = X[train_idx], y[train_idx], y_binary[train_idx]
    X_val, y_val, y_val_binary = (
        X[temp_idx][val_idx],
        y[temp_idx][val_idx],
        y_binary[temp_idx][val_idx],
    )
    X_test, y_test, y_test_binary = (
        X[temp_idx][test_idx],
        y[temp_idx][test_idx],
        y_binary[temp_idx][test_idx],
    )

    # Fit scalers ONLY on training data to prevent information leakage.
    scaler_v = StandardScaler().fit(X_train[:, :, 0].reshape(-1, 1))
    scaler_c = StandardScaler().fit(X_train[:, :, 1].reshape(-1, 1))

    # Apply the fitted scalers to all datasets.
    for dset in [X_train, X_val, X_test]:
        dset[:, :, 0] = scaler_v.transform(dset[:, :, 0].reshape(-1, 1)).reshape(
            dset.shape[0], dset.shape[1]
        )
        dset[:, :, 1] = scaler_c.transform(dset[:, :, 1].reshape(-1, 1)).reshape(
            dset.shape[0], dset.shape[1]
        )

    # Extract wavelet features for all datasets.
    X_wavelet_train = extract_wavelet_features(
        X_train, cfg.wavelet_family, cfg.wavelet_level
    )
    X_wavelet_val = extract_wavelet_features(
        X_val, cfg.wavelet_family, cfg.wavelet_level
    )
    X_wavelet_test = extract_wavelet_features(
        X_test, cfg.wavelet_family, cfg.wavelet_level
    )

    # Package data into dictionaries for clarity.
    train_data = {
        "X": X_train,
        "X_wavelet": X_wavelet_train,
        "y": y_train,
        "y_binary": y_train_binary,
    }
    val_data = {
        "X": X_val,
        "X_wavelet": X_wavelet_val,
        "y": y_val,
        "y_binary": y_val_binary,
    }
    test_data = {
        "X": X_test,
        "X_wavelet": X_wavelet_test,
        "y": y_test,
        "y_binary": y_test_binary,
    }
    artifacts = {"scaler_voltage": scaler_v, "scaler_current": scaler_c}

    return train_data, val_data, test_data, artifacts


def create_tf_dataset(data, batch_size, is_training=True):
    dummy_attention = np.zeros((len(data["X"]), data["X"].shape[1]))
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"time_series_input": data["X"], "wavelet_input": data["X_wavelet"]},
            {
                "binary_output": data["y_binary"],
                "multiclass_output": data["y"],
                "attention_output": dummy_attention,
            },
        )
    )
    if is_training:
        dataset = (
            dataset.shuffle(buffer_size=len(data["X"]))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# =============================================================================
# 3. ADVANCED LOSS FUNCTION
# =============================================================================
def categorical_focal_loss(alpha: list, gamma: float = 2.0):
    """
    Implements the Focal Loss function for multiclass classification.
    This loss function is highly effective for datasets with class imbalance.

    Args:
        alpha (list): A list of weights for each class.
        gamma (float): The focusing parameter.

    Returns:
        A Keras loss function.
    """
    alpha = np.array(alpha, dtype=np.float32)

    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=len(alpha))
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(K.sum(loss, axis=-1))

    return focal_loss_fixed


# =============================================================================
# 4. MODEL ARCHITECTURE
# =============================================================================
def create_transformer_encoder_block(inputs, d_model, num_heads, ff_dim, dropout_rate):
    """Creates a single Transformer Encoder block with an attention score output."""
    attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    # The layer returns attention scores which we will use for visualization.
    attention_output, attention_scores = attention_layer(
        inputs, inputs, return_attention_scores=True
    )
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(
        inputs + attention_output
    )
    ffn = layers.Dense(ff_dim, activation="relu")(attention_output)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn)
    return output, attention_scores


def build_transformer_fault_model(
    ts_shape: tuple, wavelet_shape: int, cfg: Config, class_weights: dict
):
    """
    Builds the final dual-input, triple-output Transformer model.
    """
    time_series_input = layers.Input(shape=ts_shape, name="time_series_input")
    wavelet_input = layers.Input(shape=(wavelet_shape,), name="wavelet_input")

    # Time-series branch with Transformer encoders
    x = layers.Dense(cfg.d_model)(time_series_input)
    # Add positional encoding to give the model information about time step order.
    pos_enc = layers.Embedding(input_dim=ts_shape[0], output_dim=cfg.d_model)(
        tf.range(start=0, limit=ts_shape[0], delta=1)
    )
    x += pos_enc
    attention_scores = None
    for _ in range(cfg.num_transformer_blocks):
        x, attention_scores = create_transformer_encoder_block(
            x, cfg.d_model, cfg.num_heads, cfg.ff_dim, cfg.transformer_dropout
        )

    # This special output layer makes attention scores accessible for XAI.
    attention_output = layers.Lambda(lambda t: t, name="attention_output")(
        attention_scores
    )
    ts_features = layers.GlobalAveragePooling1D()(x)

    # Wavelet feature branch
    wavelet_branch = layers.Dense(128, activation="relu")(wavelet_input)
    wavelet_branch = layers.BatchNormalization()(wavelet_branch)
    wavelet_branch = layers.Dropout(0.3)(wavelet_branch)
    wavelet_branch = layers.Dense(64, activation="relu")(wavelet_branch)

    # Fuse the two branches
    fused = layers.Concatenate()([ts_features, wavelet_branch])

    # Classification head
    head = layers.Dense(128, activation="relu")(fused)
    head = layers.BatchNormalization()(head)
    head = layers.Dropout(0.5)(head)
    head = layers.Dense(64, activation="relu")(head)
    binary_output = layers.Dense(1, activation="sigmoid", name="binary_output")(head)
    multiclass_output = layers.Dense(
        len(cfg.label_map), activation="softmax", name="multiclass_output"
    )(head)

    model = models.Model(
        inputs=[time_series_input, wavelet_input],
        outputs=[binary_output, multiclass_output, attention_output],
    )

    # Select the multiclass loss function based on the configuration.
    multiclass_loss = "sparse_categorical_crossentropy"
    if cfg.use_focal_loss:
        alpha = [class_weights.get(i, 1.0) for i in sorted(class_weights.keys())]
        multiclass_loss = categorical_focal_loss(alpha=alpha)
        logging.info("Using Focal Loss for multiclass output.")

    model.compile(
        optimizer=optimizers.Adam(cfg.initial_learning_rate),
        loss={
            "binary_output": "binary_crossentropy",
            "multiclass_output": multiclass_loss,
            "attention_output": None,
        },  # No loss for attention output
        metrics={"binary_output": "accuracy", "multiclass_output": "accuracy"},
    )
    return model


# =============================================================================
# 5. EXPLAINABLE AI (XAI) MODULE
# =============================================================================
def run_shap_analysis(
    model: models.Model, train_data: Dict, test_data: Dict, cfg: Config
):
    """Generates and saves a SHAP summary plot for feature importance."""
    logging.info("Running SHAP analysis...")
    background_ts = train_data["X"][: cfg.shap_background_samples]
    background_wv = train_data["X_wavelet"][: cfg.shap_background_samples]

    # Create a sub-model that only outputs the multiclass predictions for SHAP.
    explainer_model = models.Model(
        inputs=model.inputs, outputs=[model.get_layer("multiclass_output").output]
    )
    explainer = shap.GradientExplainer(explainer_model, [background_ts, background_wv])

    shap_values = explainer.shap_values([test_data["X"], test_data["X_wavelet"]])

    # Reshape time-series data for a combined plot
    X_test_combined = np.concatenate(
        [test_data["X"].reshape(len(test_data["X"]), -1), test_data["X_wavelet"]],
        axis=1,
    )

    plt.figure()
    shap.summary_plot(
        np.mean(np.abs(shap_values), axis=1),
        X_test_combined,
        max_display=20,
        show=False,
    )
    plt.title("SHAP Feature Importance (Multiclass Output)")
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()
    mlflow.log_artifact("shap_summary.png")
    logging.info("SHAP summary plot saved and logged to MLflow.")


def visualize_attention_heatmap(
    trained_model: models.Model,
    sample_ts: np.ndarray,
    sample_wv: np.ndarray,
    filename="attention_heatmap.png",
):
    """Generates and saves an attention heatmap for a single test sample."""
    logging.info("Generating attention visualization...")
    # Create a sub-model that only outputs the attention scores.
    attention_model = models.Model(
        inputs=trained_model.inputs,
        outputs=trained_model.get_layer("attention_output").output,
    )
    attention_scores = attention_model.predict(
        [np.expand_dims(sample_ts, 0), np.expand_dims(sample_wv, 0)]
    )

    # Average attention scores across all heads for visualization.
    avg_attention = np.mean(attention_scores[0], axis=0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax1.plot(sample_ts[:, 0], label="Voltage", c="b")
    ax1.set_ylabel("Voltage (Scaled)")
    ax1.legend(loc="upper left")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(sample_ts[:, 1], label="Current", c="r")
    ax1_twin.set_ylabel("Current (Scaled)")
    ax1_twin.legend(loc="upper right")
    ax1.set_title("Input Waveform & Model Attention")
    ax1.grid(True, linestyle="--", alpha=0.6)

    im = ax2.imshow(np.expand_dims(avg_attention, 0), cmap="viridis", aspect="auto")
    ax2.set_xlabel("Time Step")
    ax2.set_yticks([])
    cbar = fig.colorbar(im, ax=ax2, orientation="horizontal", fraction=0.05, pad=0.25)
    cbar.set_label("Attention Score (Higher is more focus)")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    mlflow.log_artifact(filename)
    logging.info("Attention heatmap saved and logged to MLflow.")


# =============================================================================
# 6. TRAINING AND EVALUATION
# =============================================================================
def train_and_evaluate(cfg: Config):
    """
    The main function that orchestrates the entire ML pipeline.
    """
    # --- 6.1 Setup ---
    set_seeds(cfg.seed)
    train_data, val_data, test_data, artifacts = preprocess_and_split_data(cfg)

    # --- 6.2 Data Pipeline ---
    # Convert NumPy arrays to efficient tf.data.Dataset objects for training.
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                {
                    "time_series_input": train_data["X"],
                    "wavelet_input": train_data["X_wavelet"],
                },
                {
                    "binary_output": train_data["y_binary"],
                    "multiclass_output": train_data["y"],
                    "attention_output": np.zeros(
                        (len(train_data["X"]), train_data["X"].shape[1])
                    ),
                },
            )
        )
        .shuffle(buffer_size=len(train_data["X"]))
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                {
                    "time_series_input": val_data["X"],
                    "wavelet_input": val_data["X_wavelet"],
                },
                {
                    "binary_output": val_data["y_binary"],
                    "multiclass_output": val_data["y"],
                    "attention_output": np.zeros(
                        (len(val_data["X"]), val_data["X"].shape[1])
                    ),
                },
            )
        )
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                {
                    "time_series_input": test_data["X"],
                    "wavelet_input": test_data["X_wavelet"],
                },
                {
                    "binary_output": test_data["y_binary"],
                    "multiclass_output": test_data["y"],
                    "attention_output": np.zeros(
                        (len(test_data["X"]), test_data["X"].shape[1])
                    ),
                },
            )
        )
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # --- 6.3 Class Imbalance Handling ---
    # Compute class weights to penalize the model more for mistakes on rare classes.
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_data["y"]), y=train_data["y"]
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    logging.info(f"Computed class weights for loss: {class_weights_dict}")

    # --- 6.4 Experiment Tracking Setup ---
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    with mlflow.start_run() as run:
        logging.info(f"MLflow Run Started: {run.info.run_id}")
        mlflow.log_params(vars(cfg))
        mlflow.log_dict(class_weights_dict, "class_weights.json")

        # --- 6.5 Model Training ---
        logging.info("Building and training the final model...")
        model = build_transformer_fault_model(
            train_data["X"].shape[1:],
            train_data["X_wavelet"].shape[1],
            cfg,
            class_weights_dict,
        )
        model.summary()

        # --- 6.5 Define callbacks for smarter training ---
        training_callbacks = [
            # Stop training if validation accuracy doesn't improve. Restore the best weights found.
            callbacks.EarlyStopping(
                monitor="val_multiclass_output_accuracy",
                patience=cfg.early_stopping_patience,
                mode="max",
                restore_best_weights=True,
            ),
            # Reduce learning rate when performance plateaus.
            callbacks.ReduceLROnPlateau(
                monitor="val_multiclass_output_accuracy",
                factor=0.2,
                patience=cfg.lr_scheduler_patience,
                min_lr=1e-6,
                mode="max",
                verbose=1,
            ),
        ]

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=cfg.epochs,
            callbacks=training_callbacks,
            # Apply class weights only if not using Focal Loss (which has its own weighting).
            class_weight={"multiclass_output": class_weights_dict}
            if not cfg.use_focal_loss
            else None,
            verbose=2,
        )

        # --- 6.6 Evaluation ---
        logging.info("Evaluating final model on the unseen test set...")
        eval_results = model.evaluate(test_dataset)
        _, preds_multi, _ = model.predict(test_dataset)
        preds_multi = np.argmax(preds_multi, axis=1)

        metrics = {
            "test_loss": eval_results[0],
            "test_binary_accuracy": eval_results[3],
            "test_multiclass_accuracy": eval_results[4],
        }
        mlflow.log_metrics(metrics)
        logging.info(f"Final Test Metrics: {metrics}")

        report = classification_report(
            test_data["y"],
            preds_multi,
            target_names=cfg.label_map.keys(),
            output_dict=True,
        )
        mlflow.log_dict(report, "classification_report.json")
        logging.info(
            "Classification Report:\n"
            + classification_report(
                test_data["y"], preds_multi, target_names=cfg.label_map.keys()
            )
        )

        # --- 6.7 XAI and Artifact Logging ---
        run_shap_analysis(model, train_data, test_data, cfg)
        visualize_attention_heatmap(model, test_data["X"][0], test_data["X_wavelet"][0])

        logging.info("Saving final model and artifacts to MLflow...")
        mlflow.keras.log_model(model, "transformer_model")
        with open("scalers.pkl", "wb") as f:
            pickle.dump(artifacts, f)
        mlflow.log_artifact("scalers.pkl")

        # Log training history plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["multiclass_output_accuracy"], label="Train Acc")
        plt.plot(history.history["val_multiclass_output_accuracy"], label="Val Acc")
        plt.title("Model Accuracy")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(history.history["lr"], label="Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_history.png")
        mlflow.log_artifact("training_history.png")

        logging.info(f"--- Pipeline Complete. MLflow Run ID: {run.info.run_id} ---")


if __name__ == "__main__":
    if not os.path.exists("data"):
        logging.error(
            "FATAL: 'data' directory not found. Please place the .xlsx files in a 'data' subfolder."
        )
    else:
        config = Config()
        train_and_evaluate(config)
