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
for ii, model in enumerate(models):
    scores[model] = cosine_similarity[ii] - human_score
    means.append(
        np.mean(scores[model])
    )
    if ii==0:
        mean_val = np.array(scores[model])
    else:
        mean_val += np.array(scores[model])

scores['Average'] = mean_val/len(models)
means.append(
    np.mean(scores['Average'])
)

df = pd.DataFrame.from_dict(scores)
df = pd.melt(df,var_name='Models', value_name='Bias')

#%%
human_score = np.concatenate((human_score, human_score, human_score, human_score), axis=0)
# %%
ticksize = 30
labelsize = 30
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

sns.set_context('talk')
ax_ = sns.stripplot(x='Models', y='Bias', data=df, ax=ax, size=2, color='b', legend=None, hue=human_score)

ax.scatter(
    x=np.arange(len(means)),  # positions for each category
    y=means,
    color="red",
    marker="D",
    s=70,
    zorder=3,
    label="Mean"
)
ax.hlines(0, 0, 3, linestyles='--',colors='k')

ax_.set_ylabel('Bias', fontsize=labelsize)
ax_.set_xlabel('', fontsize=labelsize)
ax_.set_xticklabels(
    scores.keys(),
    fontsize=labelsize,rotation=45
    )
ax_.set_yticks([-.4,0,.6])
ax_.tick_params(labelsize=ticksize)

right_side = ax_.spines["right"]
right_side.set_visible(False)
top_side = ax_.spines["top"]
top_side.set_visible(False)

plt.savefig('plots/embedding_bias.pdf')
# %%
