import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.tcn_lstm.model import TCNLSTM


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "results/tcn_lstm/normalized_dataset.csv"
MODEL_PATH = "results/tcn_lstm/best_model.pt"
OUTPUT_DIR = "results/tcn_lstm"

SEQUENCE_LENGTH = 100
TEST_BATTERY = "B0018"
BATCH_SIZE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("TCN-LSTM EVALUATION")
print("=" * 60)

print("Device       :", DEVICE)
print("Test battery :", TEST_BATTERY)


# ============================================================
# LOAD NORMALIZED DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

test_df = df[
    df["battery"] == TEST_BATTERY
].copy()

print("Test samples :", len(test_df))


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    "voltage_v",
    "current_a",
    "temperature_c"
]

target_column = "soc"


# ============================================================
# CREATE SEQUENCES
# ============================================================
#
# IMPORTANT:
# This must match BatterySequenceDataset used during training.
#
# Sequences are created independently for each
# (battery, cycle), so no sequence crosses a cycle boundary.
#
# Target is the final sample of each 100-point sequence:
#
# x = i : i+100
# y = i+99
#
# ============================================================

X_sequences = []
y_targets = []
metadata = []

for (battery, cycle), group in test_df.groupby(
    ["battery", "cycle"],
    sort=False
):

    group = group.sort_values(
        "time_s"
    ).reset_index(drop=True)

    features = group[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    targets = group[
        target_column
    ].to_numpy(
        dtype=np.float32
    )

    if len(group) < SEQUENCE_LENGTH:
        continue

    for i in range(
        len(group) - SEQUENCE_LENGTH + 1
    ):

        X_sequences.append(
            features[
                i:i + SEQUENCE_LENGTH
            ]
        )

        target_index = (
            i + SEQUENCE_LENGTH - 1
        )

        y_targets.append(
            targets[target_index]
        )

        metadata.append(
            {
                "battery": battery,
                "cycle": cycle,
                "time_s": group.loc[
                    target_index,
                    "time_s"
                ],
                "voltage_v": group.loc[
                    target_index,
                    "voltage_v"
                ],
                "current_a": group.loc[
                    target_index,
                    "current_a"
                ],
                "temperature_c": group.loc[
                    target_index,
                    "temperature_c"
                ],
                "capacity_ah": group.loc[
                    target_index,
                    "capacity_ah"
                ],
            }
        )


X_sequences = np.asarray(
    X_sequences,
    dtype=np.float32
)

y_targets = np.asarray(
    y_targets,
    dtype=np.float32
)

metadata = pd.DataFrame(
    metadata
)


print("Sequences    :", len(X_sequences))
print("Input shape  :", X_sequences.shape)
print("Target shape :", y_targets.shape)


# ============================================================
# LOAD MODEL
# ============================================================

model = TCNLSTM(
    input_size=3,
    tcn_channels=32,
    lstm_hidden_size=64,
    lstm_layers=2,
    dropout=0.2
).to(DEVICE)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()


# ============================================================
# BATCHED PREDICTION
# ============================================================

predictions_list = []


with torch.no_grad():

    for start in range(
        0,
        len(X_sequences),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_sequences)
        )

        X_batch = torch.tensor(
            X_sequences[start:end],
            dtype=torch.float32
        ).to(DEVICE)

        batch_prediction = model(
            X_batch
        )

        predictions_list.append(
            batch_prediction
            .cpu()
            .numpy()
            .reshape(-1)
        )


predictions = np.concatenate(
    predictions_list
)


print(
    "Predictions generated :",
    len(predictions)
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_targets,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_targets,
        predictions
    )
)

r2 = r2_score(
    y_targets,
    predictions
)


print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


results = metadata.copy()

results["soc_actual"] = y_targets

results["soc_predicted"] = predictions

results["soc_error"] = (
    results["soc_actual"]
    - results["soc_predicted"]
)


output_csv = (
    f"{OUTPUT_DIR}/evaluation_B0018.csv"
)


results.to_csv(
    output_csv,
    index=False
)


print("\nPredictions saved to:")
print(output_csv)


# ============================================================
# PLOT 1
# ACTUAL VS PREDICTED SOC
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    y_targets,
    label="Actual SOC"
)

plt.plot(
    predictions,
    label="Predicted SOC"
)

plt.xlabel(
    "Sample"
)

plt.ylabel(
    "SOC"
)

plt.title(
    "TCN-LSTM SOC Estimation - B0018"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


plot_path = (
    f"{OUTPUT_DIR}/soc_prediction_B0018.png"
)


plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


# ============================================================
# PLOT 2
# SOC ERROR
# ============================================================

error = (
    y_targets - predictions
)


plt.figure(
    figsize=(12, 4)
)

plt.plot(
    error
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Sample"
)

plt.ylabel(
    "SOC Error"
)

plt.title(
    "TCN-LSTM SOC Estimation Error - B0018"
)

plt.grid(True)

plt.tight_layout()


error_plot_path = (
    f"{OUTPUT_DIR}/soc_error_B0018.png"
)


plt.savefig(
    error_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nPlots saved to:")

print(
    plot_path
)

print(
    error_plot_path
)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)