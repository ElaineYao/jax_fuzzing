import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, export_text

proj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Read the CSV file and evaluate shape from string to tuple
results_csv = os.path.join(proj_root, 'results', 'results.csv')
df = pd.read_csv(results_csv)
df["shape"] = df["shape"].apply(eval)

# Clean up dtype to extract just 'float16', 'float32', 'float64'
df["dtype"] = df["dtype"].str.extract(r"(float\d+)")

# Optional: make dtype categorical and ordered
df["dtype"] = pd.Categorical(df["dtype"], categories=["float16", "float32", "float64"], ordered=True)

# Automatically extract parameters from the header, excluding failure_rate
parameters = [col for col in df.columns if col != "failure_rate"]

# Print the average failure rate for each parameter
grouped_means = {param: df.groupby(param)["failure_rate"].mean() for param in parameters}
for param, mean in grouped_means.items():
    print(f"Average failure rate for {param}:\n{mean}\n")

# Encode shape as area
df["shape_area"] = df["shape"].apply(lambda x: x[0] * x[1])

# Encode 'mode' (categorical) to numeric
df["mode_enc"] = df["mode"].map({"fwd": 0, "rev": 1})

# Encode 'dtype' as numeric
df["dtype_enc"] = df["dtype"].map({"float16": 0, "float32": 1, "float64": 2})

# Include all numeric params + encoded mode + shape_area
numeric_parameters = [
    param for param in parameters if pd.api.types.is_numeric_dtype(df[param])
] + ["shape_area", "mode_enc", "dtype_enc"]

# Calculate correlation with failure_rate
correlations = df[["failure_rate"] + numeric_parameters].corr()["failure_rate"].drop("failure_rate")
print("Correlations with failure_rate:")
print(correlations)

# Group by eps and dtype to visualize average failure_rate
eps_dtype_avg = df.groupby(["eps", "dtype"])["failure_rate"].mean().reset_index()
pivot = eps_dtype_avg.pivot(index="dtype", columns="eps", values="failure_rate")

# Plot heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Failure Rate by eps and dtype")
plt.xlabel("eps")
plt.ylabel("dtype")
plt.tight_layout()
plt.savefig(os.path.join(proj_root, 'results', 'failure_rate_heatmap_eps_dtype.png'))

# Group by atol and dtype to visualize average failure_rate
atol_dtype_avg = df.groupby(["atol", "dtype"])["failure_rate"].mean().reset_index()
pivot = atol_dtype_avg.pivot(index="dtype", columns="atol", values="failure_rate")

# Plot heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Failure Rate by atol and dtype")
plt.xlabel("atol")
plt.ylabel("dtype")
plt.tight_layout()
plt.savefig(os.path.join(proj_root, 'results', 'failure_rate_heatmap_atol_dtype.png'))
