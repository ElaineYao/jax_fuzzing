import pandas as pd
import sys
import os
from sklearn.tree import DecisionTreeRegressor, export_text


proj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Read the CSV file and extract the header
results_csv = os.path.join(proj_root, 'results', 'results.csv')
df = pd.read_csv(results_csv)
df["shape"] = df["shape"].apply(eval)

# Automatically extract parameters from the header, including 'shape'
parameters = [col for col in df.columns if col != "failure_rate"]

# Print the average failure rate for each parameter
grouped_means = {param: df.groupby(param)["failure_rate"].mean() for param in parameters}
for param, mean in grouped_means.items():
    print(f"Average failure rate for {param}:\n{mean}\n")

# Encode shape as area
df["shape_area"] = df["shape"].apply(lambda x: x[0] * x[1])

# Encode 'mode' (categorical) to numeric
df["mode_enc"] = df["mode"].map({"fwd": 0, "rev": 1})

# Include all numeric params + encoded mode + shape_area
numeric_parameters = [
    param for param in parameters if pd.api.types.is_numeric_dtype(df[param])
] + ["shape_area", "mode_enc"]

# Calculate correlation
correlations = df[["failure_rate"] + numeric_parameters].corr()["failure_rate"].drop("failure_rate")

print("Correlations with failure_rate:")
print(correlations)



