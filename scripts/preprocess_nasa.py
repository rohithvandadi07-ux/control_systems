from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

# Thesis reference-SOC convention
INITIAL_SOC = 1.0
COULOMBIC_EFFICIENCY = 1.0


# ============================================================
# MATLAB loader
# ============================================================

def load_battery(mat_file: Path, battery_name: str):
    data = loadmat(
        mat_file,
        squeeze_me=True,
        struct_as_record=False,
    )

    return data[battery_name]


# ============================================================
# Extract discharge cycles
# ============================================================

def extract_discharge_cycles(battery, battery_name: str):

    records = []

    for cycle_index, cycle in enumerate(battery.cycle):

        if cycle.type != "discharge":
            continue

        d = cycle.data

        time = np.asarray(d.Time, dtype=float)
        voltage = np.asarray(d.Voltage_measured, dtype=float)
        current = np.asarray(d.Current_measured, dtype=float)
        temperature = np.asarray(
            d.Temperature_measured,
            dtype=float
        )

        capacity_ah = float(d.Capacity)

        # ----------------------------------------------------
        # Remove invalid values
        # ----------------------------------------------------

        valid = (
            np.isfinite(time)
            & np.isfinite(voltage)
            & np.isfinite(current)
            & np.isfinite(temperature)
        )

        time = time[valid]
        voltage = voltage[valid]
        current = current[valid]
        temperature = temperature[valid]

        # ----------------------------------------------------
        # Remove duplicate timestamps
        # ----------------------------------------------------

        unique_time, unique_indices = np.unique(
            time,
            return_index=True
        )

        time = unique_time
        voltage = voltage[unique_indices]
        current = current[unique_indices]
        temperature = temperature[unique_indices]

        # ----------------------------------------------------
        # Sort chronologically
        # ----------------------------------------------------

        order = np.argsort(time)

        time = time[order]
        voltage = voltage[order]
        current = current[order]
        temperature = temperature[order]

        # ----------------------------------------------------
        # Reference SOC using Coulomb counting
        #
        # Thesis convention:
        # I > 0 : charging
        # I < 0 : discharging
        #
        # SOC(t) = SOC0 + eta/Qn * integral(I dt)
        # ----------------------------------------------------

        time_hours = time / 3600.0

        delta_t = np.diff(time_hours)

        current_mid = (
            current[:-1] + current[1:]
        ) / 2.0

        cumulative_charge = np.concatenate(
            [
                [0.0],
                np.cumsum(
                    current_mid * delta_t
                ),
            ]
        )

        soc = (
            INITIAL_SOC
            + (
                COULOMBIC_EFFICIENCY
                * cumulative_charge
                / capacity_ah
            )
        )

        # ----------------------------------------------------
        # Numerical safety
        # ----------------------------------------------------

        soc = np.clip(soc, 0.0, 1.0)

        # ----------------------------------------------------
        # Create records
        # ----------------------------------------------------

        for i in range(len(time)):

            records.append(
                {
                    "battery": battery_name,
                    "cycle": cycle_index + 1,
                    "time_s": time[i],
                    "voltage_v": voltage[i],
                    "current_a": current[i],
                    "temperature_c": temperature[i],
                    "capacity_ah": capacity_ah,
                    "soc": soc[i],
                }
            )

    return records


# ============================================================
# Main
# ============================================================

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_records = []

    for battery_name in BATTERIES:

        mat_file = RAW_DIR / f"{battery_name}.mat"

        print(
            f"\nProcessing {battery_name}..."
        )

        battery = load_battery(
            mat_file,
            battery_name
        )

        records = extract_discharge_cycles(
            battery,
            battery_name
        )

        all_records.extend(records)

        print(
            f"  Extracted samples: {len(records):,}"
        )

    dataframe = pd.DataFrame(
        all_records
    )

    output_file = (
        PROCESSED_DIR
        / "nasa_discharge_reference_soc.csv"
    )

    dataframe.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Total samples: {len(dataframe):,}"
    )

    print(
        f"Batteries: "
        f"{dataframe['battery'].nunique()}"
    )

    print(
        f"Output: {output_file}"
    )

    print("\nColumns:")
    print(list(dataframe.columns))


if __name__ == "__main__":
    main()
