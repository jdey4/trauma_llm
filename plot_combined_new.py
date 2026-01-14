#%%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import matplotlib as mpl
from scipy.stats import f_oneway

rcParams.update({'figure.autolayout': False})
sns.set_context('talk')

# ---------------- Figure setup ----------------
fig, ax = plt.subplots(1, 3, figsize=(22, 8), sharey=True)
ticksize = 30
labelsize = 30

titles = ['STSB', 'Biosses', 'Trauma']
models = ['gemini', 'nomic', 'openai', 'pubmedbert']

files = {
    0: [
        'stsb/gemini_text-gemini-embedding-001_dim_3072.pickle',
        'stsb/nomic_text-nomic-embed-text-v2-moe_768.pickle',
        'stsb/openai_text-embedding-3-large.pickle',
        'stsb/pubmedbert-base-embeddings_768.pickle'
    ],
    1: [
        'biosses/gemini_text-gemini-embedding-001_dim_3072.pickle',
        'biosses/nomic_text-nomic-embed-text-v2-moe_768.pickle',
        'biosses/openai_text-embedding-3-large.pickle',
        'biosses/pubmedbert-base-embeddings_768.pickle'
    ],
    2: [
        'trauma/gemini_text-gemini-embedding-001_dim_768.pickle',
        'trauma/nomic_text-nomic-embed-text-v2-moe_768.pickle',
        'trauma/openai_text-embedding-3-large.pickle',
        'trauma/pubmedbert-base-embeddings_768.pickle'
    ]
}

# Track human score range for global colorbar
global_human_min = np.inf
global_human_max = -np.inf

# ---------------- Main loop ----------------
for kk in range(3):
    cosine_similarity = []
    human_score = None

    for file in files[kk]:
        with open(file, 'rb') as f:
            data = pickle.load(f)

        cosine_similarity.append(np.array(data['Cosine similarity']))
        if human_score is None:
            human_score = np.array(data['Human score'])

    global_human_min = min(global_human_min, human_score.min())
    global_human_max = max(global_human_max, human_score.max())

    # -------- Bias per model --------
    scores = {}
    means = []
    stds = []
    ground_truth = []

    for ii, model in enumerate(models):
        scores[model] = cosine_similarity[ii] - human_score
        means.append(np.mean(scores[model]))
        stds.append(np.std(scores[model], ddof=1))
        ground_truth.extend(human_score)

    # -------- One-way ANOVA --------
    anova_inputs = [scores[m] for m in models]
    F_stat, p_value = f_oneway(*anova_inputs)

    if p_value < 1e-4:
        p_text = f"$p < 10^{{-4}}$"
    else:
        p_text = f"$p = {p_value:.3g}$"

    # -------- Long-form dataframe --------
    df = pd.DataFrame.from_dict(scores)
    df = pd.melt(df, var_name='Models', value_name='Bias')
    df.insert(2, "human score", ground_truth)

    # -------- Stripplot --------
    ax_ = sns.stripplot(
        x='Models',
        y='Bias',
        data=df,
        ax=ax[kk],
        size=5,
        hue='human score',
        palette='coolwarm',
        alpha=0.5,
        legend=False,
        order=models
    )

    # -------- Mean ± STD --------
    xpos = np.arange(len(models))
    ax[kk].errorbar(
        xpos, means, yerr=stds,
        fmt='D',
        color='red',
        ecolor='black',
        elinewidth=3,
        capsize=10,
        capthick=3,
        markersize=8,
        zorder=5,
        label='Mean ± STD' if kk == 0 else None
    )

    # -------- Formatting --------
    ax[kk].axhline(0, linestyle='--', color='k', linewidth=1.5)
    ax[kk].set_title(
        f"{titles[kk]}\nOne-way ANOVA: {p_text}",
        fontsize=labelsize
    )
    ax_.set_xlabel('')
    ax_.set_xticklabels(models, fontsize=labelsize, rotation=80)
    ax_.set_yticks([-1, 0, 0.6])
    ax_.tick_params(labelsize=ticksize)
    ax_.spines["right"].set_visible(False)
    ax_.spines["top"].set_visible(False)

ax[0].set_ylabel('Model score − Human score', fontsize=labelsize - 4)

# ---------------- External colorbar ----------------
norm = mpl.colors.Normalize(vmin=global_human_min, vmax=global_human_max)
sm = mpl.cm.ScalarMappable(norm=norm, cmap='coolwarm')
sm.set_array([])

cax = fig.add_axes([0.92, 0.18, 0.02, 0.64])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('Human Score', fontsize=labelsize)
cbar.ax.tick_params(labelsize=ticksize)

# # ---------------- Legend ----------------
# fig.legend(
#     loc='upper center',
#     bbox_to_anchor=(0.5, 0.02),
#     fontsize=labelsize,
#     ncol=1,
#     frameon=False
# )

plt.subplots_adjust(left=0.06, right=0.9, bottom=0.18, top=0.88, wspace=0.25)
plt.savefig('plots/embedding_bias_with_anova.pdf')
plt.show()
#%%
