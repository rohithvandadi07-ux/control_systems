from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BatterySequenceDataset(Dataset):
    """
    Converts battery time-series data into sliding-window sequences
    for TCN-LSTM SOC estimation.

    Inputs:
        voltage_v
        current_a
        temperature_c

    Target:
        soc
    """

    def __init__(
        self,
        csv_path,
        sequence_length=100,
        battery_ids=None,
        cycles=None,
    ):
        self.csv_path = Path(csv_path)
        self.sequence_length = sequence_length

        df = pd.read_csv(self.csv_path)

        required_columns = [
            "battery",
            "cycle",
            "time_s",
            "voltage_v",
            "current_a",
            "temperature_c",
            "soc",
        ]

        missing = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # Optional battery filtering
        if battery_ids is not None:
            df = df[df["battery"].isin(battery_ids)]

        # Optional cycle filtering
        if cycles is not None:
            df = df[df["cycle"].isin(cycles)]

        # Remove invalid rows
        df = df.dropna(
            subset=[
                "voltage_v",
                "current_a",
                "temperature_c",
                "soc",
            ]
        ).copy()

        # Keep chronological order
        df = df.sort_values(
            ["battery", "cycle", "time_s"]
        ).reset_index(drop=True)

        self.sequences = []
        self.targets = []

        feature_columns = [
            "voltage_v",
            "current_a",
            "temperature_c",
        ]

        # Create sequences independently for each battery/cycle.
        # This prevents a sequence from crossing cycle boundaries.
        for (battery, cycle), group in df.groupby(
            ["battery", "cycle"],
            sort=False,
        ):
            features = group[feature_columns].to_numpy(
                dtype=np.float32
            )

            targets = group["soc"].to_numpy(
                dtype=np.float32
            )

            if len(group) < sequence_length:
                continue

            for i in range(
                len(group) - sequence_length + 1
            ):
                x = features[
                    i:i + sequence_length
                ]

                y = targets[
                    i + sequence_length - 1
                ]

                self.sequences.append(x)
                self.targets.append(y)

        if not self.sequences:
            raise ValueError(
                "No valid sequences were created. "
                "Check sequence_length and dataset contents."
            )

        self.sequences = np.asarray(
            self.sequences,
            dtype=np.float32,
        )

        self.targets = np.asarray(
            self.targets,
            dtype=np.float32,
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        x = torch.tensor(
            self.sequences[index],
            dtype=torch.float32,
        )

        y = torch.tensor(
            self.targets[index],
            dtype=torch.float32,
        )

        return x, y


if __name__ == "__main__":
    dataset = BatterySequenceDataset(
        "data/processed/nasa_discharge_reference_soc.csv",
        sequence_length=100,
    )

    print("=" * 60)
    print("Battery Sequence Dataset")
    print("=" * 60)

    print(f"Number of sequences : {len(dataset)}")
    print(f"Sequence shape      : {dataset[0][0].shape}")
    print(f"Target shape        : {dataset[0][1].shape}")

    print()
    print("First sequence:")
    print(dataset[0][0])

    print()
    print("First target SOC:")
    print(dataset[0][1].item())

    print("=" * 60)
