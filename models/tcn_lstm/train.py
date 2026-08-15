from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.tcn_lstm.dataset import BatterySequenceDataset
from models.tcn_lstm.model import TCNLSTM


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/nasa_discharge_reference_soc.csv"

TRAIN_BATTERIES = ["B0005", "B0006", "B0007"]
TEST_BATTERIES = ["B0018"]

SEQUENCE_LENGTH = 100

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3

MODEL_DIR = Path("results/tcn_lstm")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_model.pt"
LOSS_PATH = MODEL_DIR / "training_history.csv"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("TCN-LSTM TRAINING")
print("=" * 60)

print(f"Device          : {device}")
print(f"Train batteries : {TRAIN_BATTERIES}")
print(f"Test battery    : {TEST_BATTERIES}")
print(f"Sequence length : {SEQUENCE_LENGTH}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Epochs          : {EPOCHS}")
print(f"Learning rate   : {LEARNING_RATE}")


# ============================================================
# Load raw processed data
# ============================================================

df = pd.read_csv(DATA_PATH)

feature_columns = [
    "voltage_v",
    "current_a",
    "temperature_c",
]

target_column = "soc"


# ============================================================
# Calculate normalization statistics
# USING TRAINING BATTERIES ONLY
# ============================================================

train_df = df[
    df["battery"].isin(TRAIN_BATTERIES)
].copy()

feature_mean = train_df[
    feature_columns
].mean()

feature_std = train_df[
    feature_columns
].std()

# Prevent division by zero
feature_std = feature_std.replace(0, 1.0)

print("\nTraining normalization statistics:")

for feature in feature_columns:
    print(
        f"{feature:20s} "
        f"mean={feature_mean[feature]:.6f} "
        f"std={feature_std[feature]:.6f}"
    )


# ============================================================
# Normalize complete dataframe
# ============================================================

df_normalized = df.copy()

df_normalized[feature_columns] = (
    df_normalized[feature_columns] - feature_mean
) / feature_std


# ============================================================
# Save temporary normalized dataset
# ============================================================

normalized_path = MODEL_DIR / "normalized_dataset.csv"

df_normalized.to_csv(
    normalized_path,
    index=False,
)

print(
    f"\nNormalized dataset saved to: "
    f"{normalized_path}"
)


# ============================================================
# Create datasets
# ============================================================

train_dataset = BatterySequenceDataset(
    normalized_path,
    sequence_length=SEQUENCE_LENGTH,
    battery_ids=TRAIN_BATTERIES,
)

test_dataset = BatterySequenceDataset(
    normalized_path,
    sequence_length=SEQUENCE_LENGTH,
    battery_ids=TEST_BATTERIES,
)

print("\nDataset sizes:")
print(f"Training sequences : {len(train_dataset)}")
print(f"Testing sequences  : {len(test_dataset)}")


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# Model
# ============================================================

model = TCNLSTM(
    input_size=3,
    tcn_channels=32,
    lstm_hidden_size=64,
    lstm_layers=2,
    dropout=0.2,
).to(device)

print("\nModel parameters:")
print(
    sum(
        parameter.numel()
        for parameter in model.parameters()
    )
)


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# Training
# ============================================================

best_test_loss = float("inf")

history = []


for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # Training mode
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_samples = 0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        predictions = model(x)

        loss = criterion(
            predictions,
            y,
        )

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = x.size(0)

        train_loss += (
            loss.item() * batch_size
        )

        train_samples += batch_size

    train_loss /= train_samples


    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    test_loss = 0.0
    test_samples = 0

    predictions_all = []
    targets_all = []

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(device)
            y = y.to(device)

            predictions = model(x)

            loss = criterion(
                predictions,
                y,
            )

            batch_size = x.size(0)

            test_loss += (
                loss.item() * batch_size
            )

            test_samples += batch_size

            predictions_all.append(
                predictions.cpu().numpy()
            )

            targets_all.append(
                y.cpu().numpy()
            )

    test_loss /= test_samples


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    predictions_all = np.concatenate(
        predictions_all
    )

    targets_all = np.concatenate(
        targets_all
    )

    mae = np.mean(
        np.abs(
            predictions_all -
            targets_all
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                predictions_all -
                targets_all
            ) ** 2
        )
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if test_loss < best_test_loss:

        best_test_loss = test_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "sequence_length": SEQUENCE_LENGTH,
                "feature_columns": feature_columns,
                "feature_mean": feature_mean.to_dict(),
                "feature_std": feature_std.to_dict(),
                "train_batteries": TRAIN_BATTERIES,
                "test_batteries": TEST_BATTERIES,
            },
            BEST_MODEL_PATH,
        )

        best_marker = "  <-- BEST"

    else:

        best_marker = ""


    # --------------------------------------------------------
    # Record history
    # --------------------------------------------------------

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "mae": mae,
            "rmse": rmse,
        }
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Test Loss: {test_loss:.6f} | "
        f"MAE: {mae:.6f} | "
        f"RMSE: {rmse:.6f}"
        f"{best_marker}"
    )


# ============================================================
# Save training history
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    LOSS_PATH,
    index=False,
)


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Best test loss : {best_test_loss:.6f}")
print(f"Best model     : {BEST_MODEL_PATH}")
print(f"History        : {LOSS_PATH}")

print("=" * 60)
