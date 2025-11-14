import os
import json
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

columns = ['load', 'solar', 'price']

class DatasetLoader:
    def __init__(
        self,
        csv_path: str = "data/dataset.csv",
        scalers_dir: str = "data/scalers",
        validate_columns: Optional[List[str]] = None,
    ):
        """
        Args:
            csv_path: path to the CSV dataset (expects columns: load, solar, price)
            scalers_dir: directory where scalers (min/max) will be saved/loaded
            validate_columns: list of expected columns; defaults to DEFAULT_COLUMNS
        """
        self.csv_path = csv_path
        self.scalers_dir = scalers_dir
        self.validate_columns = validate_columns or columns

        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        os.makedirs(self.scalers_dir, exist_ok=True)

        self.raw_df = self._load_csv()
        self._validate()
        self._add_datetime_features()
        # set after creating/scaling
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.normalized_df: Optional[pd.DataFrame] = None

    # Loading & validation

    def _load_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        return df

    def _validate(self):
        missing = [c for c in self.validate_columns if c not in self.raw_df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        # Ensure numeric types for expected columns
        for c in self.validate_columns:
            if not pd.api.types.is_numeric_dtype(self.raw_df[c]):
                # try converting
                self.raw_df[c] = pd.to_numeric(self.raw_df[c], errors="raise")

    def _add_datetime_features(self):
        # If there's no datetime index, create a simple hour index based on rows
        n = len(self.raw_df)
        # create hour of day and day index (useful for cyclic features)
        hours = np.arange(n) % 24
        days = np.arange(n) // 24
        self.raw_df["hour"] = hours
        self.raw_df["day"] = days
        # cyclical features for hour (useful for ML/RL)
        self.raw_df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        self.raw_df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    def fit_and_save_scalers(self):
        """
        Fit MinMax scalers for each numeric column we want to normalize,
        and save them to scalers_dir as json files (min,max).
        """
        cols_to_scale = self._scaling_columns()
        for col in cols_to_scale:
            vals = self.raw_df[[col]].values.astype(float)
            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            scaler.fit(vals)
            self.scalers[col] = scaler
            # save min/max
            meta = {"min_": float(scaler.data_min_[0]), "max_": float(scaler.data_max_[0])}
            with open(os.path.join(self.scalers_dir, f"{col}_scaler.json"), "w") as fh:
                json.dump(meta, fh)

    def load_scalers(self):
        """
        Load scaler metadata (.json) files from scalers_dir and construct MinMaxScaler objects.
        """
        cols_to_scale = self._scaling_columns()
        for col in cols_to_scale:
            path = os.path.join(self.scalers_dir, f"{col}_scaler.json")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Scaler metadata missing for '{col}' at {path}")
            with open(path, "r") as fh:
                meta = json.load(fh)
            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            # sklearn's MinMaxScaler expects arrays; we set attributes manually
            scaler.data_min_ = np.array([meta["min_"]])
            scaler.data_max_ = np.array([meta["max_"]])
            scaler.data_range_ = scaler.data_max_ - scaler.data_min_
            scaler.scale_ = 1.0 / (scaler.data_range_)
            scaler.min_ = -scaler.data_min_ * scaler.scale_
            self.scalers[col] = scaler

    def _scaling_columns(self) -> List[str]:
        # We scale all main numeric columns plus maybe derived features if desired.
        # For env inputs we typically scale: load, solar, price
        return list(self.validate_columns)

    # Normalization helpers
    def normalize_df(self) -> pd.DataFrame:
        """
        Return a DataFrame with the scaled columns replaced by normalized values [0,1].
        Non-scaled columns (hour, hour_sin, hour_cos, day) are left as-is.
        """
        if not self.scalers:
            # First attempt to load scalers; if not present, compute them.
            try:
                self.load_scalers()
            except FileNotFoundError:
                self.fit_and_save_scalers()

        df = self.raw_df.copy()
        for col, scaler in self.scalers.items():
            vals = df[[col]].values.astype(float)
            # transform using stored min/max (safe even if scaler was manually constructed)
            scaled = (vals - scaler.data_min_) / scaler.data_range_
            scaled = np.clip(scaled, 0.0, 1.0)
            df[col] = scaled.flatten()
        self.normalized_df = df
        return df

    def get_normalized_df(self) -> pd.DataFrame:
        if self.normalized_df is None:
            return self.normalize_df()
        return self.normalized_df


    # Access helpers for env/ML
    def n_timesteps(self) -> int:
        return len(self.raw_df)

    def get_row(self, t: int) -> pd.Series:
        """
        Return raw row at global timestep t (un-normalized).
        """
        if t < 0 or t >= self.n_timesteps():
            raise IndexError("t out of range")
        return self.raw_df.iloc[t]

    def get_state_at(self, t: int, include_hour_cycle: bool = True) -> np.ndarray:
        """
        Return a normalized state vector for timestep t.
        By default includes: [load, solar, price, battery_placeholder(0), hour_sin, hour_cos]
        Battery is included as a placeholder (0) so the env can append actual SoC.
        """
        df = self.get_normalized_df()
        row = df.iloc[t]
        state = [row["load"], row["solar"], row["price"]]
        if include_hour_cycle:
            state += [row["hour_sin"], row["hour_cos"]]
        # battery placeholder
        state = np.array(state, dtype=float)
        return state

    def get_windows(self, window_size: int = 24, features: Optional[List[str]] = None) -> np.ndarray:
        """
        Create sliding windows over the normalized dataframe.

        Returns:
            numpy array of shape (n_windows, window_size, n_features)
        """
        df = self.get_normalized_df()
        features = features or (self._scaling_columns() + ["hour_sin", "hour_cos"])
        arr = df[features].values.astype(float)
        n = arr.shape[0]
        if n < window_size:
            raise ValueError("Not enough timesteps to make a single window")
        n_windows = n - window_size + 1
        windows = np.stack([arr[i : i + window_size] for i in range(n_windows)], axis=0)
        return windows

    def save_normalized_csv(self, out_path: str = "data/dataset_normalized.csv"):
        df = self.get_normalized_df()
        df.to_csv(out_path, index=False)
        return out_path



# CLI / quick test
def _quick_test(csv_path="data/dataset.csv"):
    print("Quick test for DatasetLoader")
    loader = DatasetLoader(csv_path=csv_path, scalers_dir="data/scalers_test")
    print(f"Loaded rows: {loader.n_timesteps()}")
    print("Fitting scalers and normalizing...")
    loader.fit_and_save_scalers()
    dfn = loader.get_normalized_df()
    print("Normalized sample (first 5 rows):")
    print(dfn.head(5))
    print("Window shapes (24h):", loader.get_windows(24).shape)
    print("State at t=100 (normalized):", loader.get_state_at(100))
    saved = loader.save_normalized_csv("data/dataset_normalized_test.csv")
    print("Saved normalized CSV to:", saved)


if __name__ == "__main__":
    # run quick test if invoked as script
    _quick_test()