from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    classification_report,
)
DATA_PATH = Path("country_wise_latest.csv")
data = pd.read_csv(DATA_PATH)

default_countries = ["India", "United States", "Brazil", "United Kingdom"]

print("\n--- Dataset Snapshot ---")
print(data.head())

print("\n--- Columns ---")
print(data.columns.tolist())

print("\n--- Statistical Summary ---")
print(data.describe())


# Linear Regression Model

print("\n--- Linear Regression: Predicting Deaths ---")

X = data[["Confirmed", "Recovered", "Active"]]
y = data["Deaths"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.3f}")

print("\nSample Predictions:")
print(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).head())

# Random Forest 
print("\n--- Random Forest Classification ---")

if "VaccinationRate" not in data.columns:
    np.random.seed(42)
    data["VaccinationRate"] = np.random.rand(len(data))

if "LockdownFlag" not in data.columns:
    data["LockdownFlag"] = np.random.randint(0, 2, size=len(data))

# Create binary target 
data["FatalityRate"] = data["Deaths"] / (data["Confirmed"] + 1e-6)
median_threshold = data["FatalityRate"].median()
data["HighFatality"] = (data["FatalityRate"] > median_threshold).astype(int)

X_cls = data[["VaccinationRate", "LockdownFlag"]]
y_cls = data["HighFatality"]

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_cls = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred_cls)
prec = precision_score(y_test, y_pred_cls)

print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_cls))

# --- Feature--
plt.figure()
sns.barplot(x=X_cls.columns, y=rf.feature_importances_, color="skyblue")
plt.title("Random Forest Feature Importance")
plt.ylabel("Importance")
plt.show()

# --- Predictions vs Actual ---
plt.figure()
plt.scatter(range(len(y_test)), y_test, label="Actual", marker="o")
plt.scatter(range(len(y_pred_cls)), y_pred_cls, label="Predicted", marker="x")
plt.title("Random Forest: Actual vs Predicted Fatality Classes")
plt.xlabel("Test Sample Index")
plt.ylabel("High Fatality (0=Low, 1=High)")
plt.legend()
plt.show()


# Correlation & Visualization

print("\n--- Correlation Analysis ---")
corr = data.corr(numeric_only=True)
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plots
plt.figure()
sns.scatterplot(data=data, x="Confirmed", y="Deaths", hue="WHO Region")
plt.title("Confirmed vs Deaths by WHO Region")
plt.show()

plt.figure()
sns.scatterplot(data=data, x="Confirmed", y="Recovered", hue="WHO Region")
plt.title("Confirmed vs Recovered by WHO Region")
plt.show()

top10 = data.nlargest(10, "Confirmed")
plt.figure(figsize=(10, 5))
sns.barplot(data=top10, x="Country/Region", y="Confirmed", color="steelblue")
plt.title("Top 10 Countries by Confirmed Cases")
plt.xticks(rotation=45, ha="right")
plt.show()

# Vaccination vs Fatality
plt.figure()
sns.scatterplot(
    data=data,
    x="VaccinationRate",
    y="FatalityRate",
    hue="LockdownFlag",
    palette="Set1"
)
plt.title("Vaccination & Lockdown Impact on Fatality Rate")
plt.show()

def read_dataset(file_path: Path = DATA_PATH) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    return pd.read_csv(file_path)

def get_country_stats(df: pd.DataFrame, country: str) -> pd.Series:
    row = df[df["Country/Region"] == country]
    if row.empty:
        raise ValueError(f"No records for {country}")
    return row.squeeze()

def plot_top(df: pd.DataFrame, column: str, n: int, title: str):
    subset = df.nlargest(n, column)
    plt.figure()
    plt.bar(subset["Country/Region"], subset[column])
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.show()

def scatter_plot(df: pd.DataFrame, x: str, y: str, title: str):
    plt.figure()
    plt.scatter(df[x], df[y], alpha=0.7)
    if len(df) > 1:
        r = np.corrcoef(df[x], df[y])[0, 1]
        plt.title(f"{title}\nCorrelation = {r:.3f}")
    else:
        plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

def compare_countries(df: pd.DataFrame, countries: List[str], column: str):
    subset = df[df["Country/Region"].isin(countries)]
    if subset.empty:
        print("No matching countries found.")
        return
    plt.figure()
    plt.bar(subset["Country/Region"], subset[column])
    plt.xticks(rotation=30, ha="right")
    plt.title(f"Comparison of {column}")
    plt.show()

@dataclass
class CLIArgs:
    country: str
    compare: List[str]

def parse_cli(argv: Optional[List[str]] = None) -> CLIArgs:
    parser = argparse.ArgumentParser(description="COVID-19 Country Report Generator")
    parser.add_argument("--country", type=str, default="India")
    parser.add_argument("--compare", type=str, default=",".join(default_countries))
    args = parser.parse_args(argv)
    compare_list = [c.strip() for c in args.compare.split(",") if c.strip()]
    return CLIArgs(country=args.country, compare=compare_list)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_cli(argv)
    print(f"Main country: {args.country}")
    print(f"Comparison set: {', '.join(args.compare)}")

    df = read_dataset()
    stats = get_country_stats(df, args.country)
    print("\n[Country Report]")
    print(stats.to_string())

    plot_top(df, "Confirmed", 10, "Top 10 Countries by Confirmed")
    plot_top(df, "Deaths", 10, "Top 10 Countries by Deaths")
    scatter_plot(df, "Confirmed", "Deaths", "Confirmed vs Deaths")
    compare_countries(df, args.compare, "Confirmed")
    print("\nCharts generated successfully.")
    return 0


#Final Graph---
def plot_summary_lines(df: pd.DataFrame, top_n: int = 10):
    leading_countries = df.nlargest(top_n, "Confirmed")
    indicators = ["Confirmed", "Deaths", "Recovered", "Active"]
    plt.figure(figsize=(12, 6))
    for col in indicators:
        plt.plot(
            leading_countries["Country/Region"],
            leading_countries[col],
            marker="o",
            label=col
        )
    if "VaccinationRate" in leading_countries.columns:
        plt.plot(
            leading_countries["Country/Region"],
            leading_countries["VaccinationRate"] * leading_countries["Confirmed"].max(),
            marker="d",
            linestyle="--",
            label="VaccinationRate (scaled)"
        )

    if "FatalityRate" in leading_countries.columns:
        plt.plot(
            leading_countries["Country/Region"],
            leading_countries["FatalityRate"] * leading_countries["Confirmed"].max(),
            marker="s",
            linestyle=":",
            label="FatalityRate (scaled)"
        )

    plt.xticks(rotation=45, ha="right")
    plt.title(f"COVID-19 Indicators for Top {top_n} Countries")
    plt.ylabel("Counts (with scaling for rates)")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_cli(argv)
    print(f"Primary country: {args.country}")
    print(f"Comparison countries: {', '.join(args.compare)}")

    dataset = read_dataset()
    country_info = get_country_stats(dataset, args.country)

    print("\n[Country Overview]")
    print(country_info.to_string())

    plot_top(dataset, "Confirmed", 10, "Top 10 Countries by Confirmed Cases")
    plot_top(dataset, "Deaths", 10, "Top 10 Countries by Death Toll")
    scatter_plot(dataset, "Confirmed", "Deaths", "Confirmed vs Deaths Correlation")
    compare_countries(dataset, args.compare, "Confirmed")
    plot_summary_lines(dataset, top_n=10)

    print("\nAll charts created successfully.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
