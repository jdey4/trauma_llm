#%%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import matplotlib as mpl
from scipy.stats import ttest_rel
from itertools import combinations

rcParams.update({'figure.autolayout': False})
sns.set_context('talk')

# ===================== SETTINGS =====================
alpha = 0.05
models = ['gemini', 'nomic', 'openai', 'pubmedbert']
titles = ['STSB', 'Biosses', 'Trauma (expert)', 'Trauma (non-expert)']

# Bracket aesthetics (thin + nested)
BR_LW = 1.0          # thin bracket line
BR_H = 0.03          # bracket corner height (data units)
BASE_PAD = 0.06      # padding above top of dot cloud (data units)
TIER_GAP = 0.085     # vertical gap between nested brackets (data units)
STAR_FS = 18         # star font size
Q_TOP = 0.99         # use 99th percentile of dots to anchor brackets (avoids outliers)

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
    ],
    3: [
        'trauma/gemini_text-gemini-embedding-001_dim_768_non_expert.pickle',
        'trauma/nomic_text-nomic-embed-text-v2-moe_768_non_expert.pickle',
        'trauma/openai_text-embedding-3-large_non_expert.pickle',
        'trauma/pubmedbert-base-embeddings_768_non_expert.pickle'
    ]
}

# ===================== HELPERS =====================
def stars_from_p(p):
    # Stars based on RAW p-value (standard), but bracket is drawn only if Bonferroni significant.
    if p < 1e-4:
        return '****'
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    return '*'

def add_sig_bracket_onbar(ax, x1, x2, y, h, text, lw=1.0, fs=18):
    """
    Draw a bracket between categories x1 and x2.
    Put stars ON the horizontal bar (centered).
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c='k', clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y + h,                 # on the bar
        text,
        ha='center',
        va='center',
        fontsize=fs,
        clip_on=False,
        bbox=dict(facecolor='white', edgecolor='none', pad=0.15)
    )

# ===================== FIGURE SETUP =====================
fig, ax = plt.subplots(1, 4, figsize=(27, 8), sharey=True)
ticksize = 30
labelsize = 30

global_human_min = np.inf
global_human_max = -np.inf

# ===================== MAIN LOOP =====================
for kk in range(4):
    cosine_similarity = []
    human_score = None

    for file in files[kk]:
        with open(file, 'rb') as f:
            data = pickle.load(f)

        cosine_similarity.append(np.array(data['Cosine similarity']))
        if human_score is None:
            human_score = np.array(data['Human score'])

    global_human_min = min(global_human_min, float(human_score.min()))
    global_human_max = max(global_human_max, float(human_score.max()))

    # -------- Bias per model --------
    scores = {}
    means, stds = [], []
    ground_truth = []

    for ii, model in enumerate(models):
        scores[model] = cosine_similarity[ii] - human_score
        means.append(np.mean(scores[model]))
        stds.append(np.std(scores[model], ddof=1))
        ground_truth.extend(human_score)

    means = np.array(means)
    stds = np.array(stds)

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
        color='black',
        ecolor='black',
        elinewidth=2,
        capsize=8,
        capthick=2,
        markersize=10,
        zorder=5
    )

    # ==========================================================
    # Pairwise paired t-tests (6) + Bonferroni, nested brackets:
    # largest span first, then progressively smaller below.
    # ==========================================================
    pairs = list(combinations(range(len(models)), 2))
    alpha_corr = alpha / len(pairs)

    sig_pairs = []
    for i, j in pairs:
        t_stat, p = ttest_rel(scores[models[i]], scores[models[j]])  # paired
        if p < alpha_corr:
            sig_pairs.append((i, j, p))

    # Sort by span (largest first). Tie-break: smaller p first (more significant).
    sig_pairs.sort(key=lambda t: ((t[1] - t[0]), -np.log10(t[2])), reverse=True)

    # Anchor bracket block above top of the dot clouds (quantile-based, robust)
    top_cloud = max(np.quantile(scores[m], Q_TOP) for m in models)
    y_top = top_cloud + BASE_PAD

    # Draw nested: largest bracket at the top, smaller ones gradually below
    # That means we decrease y as we go down the list.
    for idx, (i, j, p) in enumerate(sig_pairs):
        y = y_top - idx * TIER_GAP
        # Safety: if nesting would dip too low into dots, instead stack upward
        if y < top_cloud + 0.02:
            y = y_top + idx * TIER_GAP

        add_sig_bracket_onbar(
            ax=ax[kk],
            x1=i,
            x2=j,
            y=y,
            h=BR_H,
            text=stars_from_p(p),
            lw=BR_LW,
            fs=STAR_FS
        )

    # Ensure brackets aren't clipped above
    if len(sig_pairs) > 0:
        ymin, ymax = ax[kk].get_ylim()
        needed_top = max(y_top + BR_H + 0.08, ymax)
        ax[kk].set_ylim(ymin, needed_top)

    # -------- Formatting --------
    ax[kk].axhline(0, linestyle='--', color='k', linewidth=1.5)
    ax[kk].set_title(titles[kk], fontsize=labelsize)
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

plt.subplots_adjust(left=0.06, right=0.9, bottom=0.18, top=0.88, wspace=0.25)
plt.savefig('plots/embedding_bias_pairwise_paired_bonferroni_nested.pdf', bbox_inches='tight')
plt.show()
#%%
