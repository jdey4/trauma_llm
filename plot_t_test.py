#%%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import matplotlib as mpl
from scipy.stats import f_oneway, ttest_rel
from itertools import combinations

rcParams.update({'figure.autolayout': False})
sns.set_context('talk')

# ====== USER TOGGLES ======
ALPHA = 0.05
SHOW_NS = True          # <- set True if you want a bracket even when not significant
Q_TOP = 0.99             # <- bracket is placed above this quantile of the dots
BRACKET_PAD = 0.03       # <- extra space above the dots (data units)
BRACKET_H = 0.05         # <- bracket height (data units)
YLIM_PAD = 0.06          # <- extra headroom to avoid clipping

# ---------------- Helper: significance bracket ----------------
def add_sig_bracket(ax, x1, x2, y, h, text, lw=2, fs=28):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c='k', clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom', fontsize=fs, clip_on=False)

def stars_from_p(p):
    if p < 1e-4: return '****'
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    return '*'

# ---------------- Figure setup ----------------
fig, ax = plt.subplots(1, 4, figsize=(27, 8), sharey=True)
ticksize = 30
labelsize = 30

titles = ['STSB', 'Biosses', 'Trauma (expert)', 'Trauma (non-expert)']
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
    ],
    3: [
        'trauma/gemini_text-gemini-embedding-001_dim_768_non_expert.pickle',
        'trauma/nomic_text-nomic-embed-text-v2-moe_768_non_expert.pickle',
        'trauma/openai_text-embedding-3-large_non_expert.pickle',
        'trauma/pubmedbert-base-embeddings_768_non_expert.pickle'
    ]
}

# Track human score range for global colorbar
global_human_min = np.inf
global_human_max = -np.inf

pairs = list(combinations(range(len(models)), 2))  # 6 pairs
m_tests = len(pairs)
alpha_corr = ALPHA / m_tests

# ---------------- Main loop ----------------
for kk in range(4):
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
        elinewidth=3,
        capsize=10,
        capthick=3,
        markersize=12,
        zorder=5
    )

    # -------- Pairwise paired t-tests + Bonferroni --------
    pair_pvals = {}
    for i, j in pairs:
        t_stat, p = ttest_rel(scores[models[i]], scores[models[j]])
        pair_pvals[(i, j)] = p

    # Best & second best = mean bias closest to 0
    order_best = np.argsort(np.abs(means))
    best_idx = int(order_best[0])
    second_best_idx = int(order_best[1])
    i, j = sorted([best_idx, second_best_idx])

    p_raw = pair_pvals[(i, j)]
    is_sig = (p_raw < alpha_corr)

    # -------- Bracket position: above the DOT CLOUDS (quantile-based) --------
    if is_sig or SHOW_NS:
        # compute top of the plotted distributions for these two groups
        top_i = np.quantile(scores[models[i]], Q_TOP)
        top_j = np.quantile(scores[models[j]], Q_TOP)
        y_base = max(top_i, top_j) + BRACKET_PAD

        text = stars_from_p(p_raw) if is_sig else "n.s."
        add_sig_bracket(ax[kk], i, j, y_base, BRACKET_H, text)

        # ensure not clipped
        ymin, ymax = ax[kk].get_ylim()
        needed_top = y_base + BRACKET_H + YLIM_PAD
        if needed_top > ymax:
            ax[kk].set_ylim(ymin, needed_top)

    # -------- Formatting --------
    ax[kk].axhline(0, linestyle='--', color='k', linewidth=1.5)
    ax[kk].set_title(f"{titles[kk]}", fontsize=labelsize)
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
plt.savefig('plots/embedding_bias_pairwise_paired_bonferroni_best_vs_second_quantile_bracket.pdf', bbox_inches='tight')
plt.show()
#%%
