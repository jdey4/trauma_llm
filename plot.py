#%%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
# %%
files = ['gemini_text-gemini-embedding-001.pickle',
         'nomic_text-nomic-embed-text-v1.pickle',
         'openai_text-embedding-3-large.pickle']
models = ['gemini', 'nomic', 'openai']
cosine_similarity = []

for file in files:
    with open(file, 'rb') as f:
        data = pickle.load(f)

    cosine_similarity.append(
        np.array(data['Cosine similarity'])
    )

human_score = np.array(data['Human score'])
# %%
scores = {}
means = []
ground_truth = []
for ii, model in enumerate(models):
    scores[model] = cosine_similarity[ii] - human_score
    means.append(
        np.mean(scores[model])
    )
    if ii==0:
        mean_val = np.array(scores[model])
    else:
        mean_val += np.array(scores[model])
    ground_truth.extend(human_score)

scores['Average'] = mean_val/len(models)
means.append(
    np.mean(scores['Average'])
)
ground_truth.extend(human_score)

df = pd.DataFrame.from_dict(scores)
df = pd.melt(df,var_name='Models', value_name='Bias')
df.insert(2, "human score", ground_truth)

# %%
ticksize = 30
labelsize = 30

# ---- Add background heatmap ----
categories = list(scores.keys())  # ['gemini', 'nomic', 'openai', 'Average']
y_bins = np.linspace(df['Bias'].min(), df['Bias'].max(), 50)  # vertical resolution
heatmap_data = np.full((len(y_bins)-1, len(categories)), np.nan)

# Draw narrow background column for each category
for xi, cat in enumerate(categories):
    # Select data for this category
    mask = df['Models'] == cat
    vals = df.loc[mask, 'Bias']
    hues = df.loc[mask, 'human score']

    # Bin the y-values
    y_bins = np.linspace(df['Bias'].min(), df['Bias'].max(), 50)
    bin_idx = np.digitize(vals, y_bins) - 1
    col_data = np.full((len(y_bins)-1, 1), np.nan)
    for bi in range(len(y_bins)-1):
        bin_vals = hues[bin_idx == bi]
        if len(bin_vals) > 0:
            col_data[bi, 0] = np.mean(bin_vals)

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.set_context('talk')

# Draw heatmap behind the stripplot
im = ax.imshow(
        np.flipud(col_data),
        extent=[xi - 0.0004, xi + 0.0004, .001, .001],
        aspect='auto',
        cmap='coolwarm',
        alpha=0.3,
        zorder=0
    )

# ---- Original stripplot ----
ax_ = sns.stripplot(
    x='Models', y='Bias',
    data=df,
    ax=ax, size=2, hue='human score',
    palette='coolwarm',
    legend=None
)

# Mean markers
ax.scatter(
    x=np.arange(len(means)),
    y=means,
    color="red",
    marker="D",
    s=70,
    zorder=3,
    label="Mean"
)

# Formatting
ax.hlines(0, 0, 3, linestyles='--', colors='k')
ax_.set_ylabel('Bias', fontsize=labelsize)
ax_.set_xlabel('', fontsize=labelsize)
ax_.set_xticklabels(categories, fontsize=labelsize, rotation=45)
ax_.set_yticks([-.4, 0, .6])
ax_.tick_params(labelsize=ticksize)
ax_.spines["right"].set_visible(False)
ax_.spines["top"].set_visible(False)

# Optional: colorbar for heatmap
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Human Score')

plt.savefig('plots/embedding_bias_with_heatmap.pdf')

# %%
