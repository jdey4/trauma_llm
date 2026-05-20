#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Load file
file_path = "trauma_data/clinical survey responses_ 3-19-26.xlsx"
df = pd.read_excel(file_path, sheet_name="Experts")

# Extract all clinician response columns
prefix = "How would you personally respond to this case?"
response_cols = [c for c in df.columns if c.startswith(prefix)]

# Flatten all responses into one list
responses = []
for col in response_cols:
    responses.extend(df[col].dropna().astype(str).tolist())

# Compute character lengths
lengths = [len(r.strip()) for r in responses if r.strip() != ""]

# Plot histogram
sns.set_context("talk")
plt.figure(figsize=(8,5))
sns.histplot(lengths, bins=50)

plt.xlabel("Response length (characters)")
plt.ylabel("Count")
plt.title("Distribution of Expert Response Lengths")

plt.tight_layout()
plt.savefig("expert_response_length_hist.png", dpi=300)
plt.show()
# %%
