#%%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import ttest_rel
from itertools import combinations

rcParams.update({'figure.autolayout': False})
sns.set_context('talk')

# ===================== SETTINGS =====================
alpha = 0.05
models = ['gemini', 'nomic', 'openai', 'pubmedbert']
titles = ['General', 'Bio-specific', 'Trauma (expert)', 'Trauma (non-expert)']

# Bracket aesthetics
BR_LW = 1.0
BR_H = 0.03
BASE_PAD = 0.06
TIER_GAP = 0.085
STAR_FS = 18
Q_TOP = 0.99

# Aggregated dots settings
N_DOTS = 300
SUB_N = 10
N_BINS = 6

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
    if p < 1e-4: return '****'
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    return '*'

def add_sig_bracket_onbar(ax, x1, x2, y, h, text, lw=1.0, fs=18):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c='k', clip_on=False)
    ax.text((x1+x2)/2, y+h, text, ha='center', va='center',
            fontsize=fs, clip_on=False,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.15))

def stratified_uniform_subsample_indices(human_score, n_total=30, n_bins=6, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    h = np.asarray(human_score)
    N = len(h)

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(h, qs)
    edges = np.unique(edges)

    if len(edges) < 2:
        return rng.choice(np.arange(N), size=n_total, replace=(N < n_total))

    bin_ids = np.digitize(h, edges[1:-1], right=True)

    per_bin = n_total // n_bins
    remainder = n_total - per_bin * n_bins

    idxs = []
    for b in range(n_bins):
        pool = np.where(bin_ids == b)[0]
        k = per_bin + (1 if b < remainder else 0)

        if len(pool) == 0:
            pick = rng.choice(np.arange(N), size=k, replace=(N < k))
        else:
            pick = rng.choice(pool, size=k, replace=(len(pool) < k))
        idxs.append(pick)

    return np.concatenate(idxs)

def make_shared_aggregated_dots(cos_sims_by_model, human_score,
                                n_dots=300, subsample_n=30,
                                n_bins=6, seed=0):

    rng = np.random.default_rng(seed)
    human_score = np.asarray(human_score)

    bias_dots_by_model = {k: [] for k in cos_sims_by_model.keys()}

    for d in range(n_dots):
        idx = stratified_uniform_subsample_indices(
            human_score, n_total=subsample_n,
            n_bins=n_bins, rng=rng)

        for model, cos_sim in cos_sims_by_model.items():
            bias = (cos_sim[idx] - human_score[idx]).mean()
            bias_dots_by_model[model].append(bias)

    for k in bias_dots_by_model:
        bias_dots_by_model[k] = np.array(bias_dots_by_model[k])

    return bias_dots_by_model

# ===================== FIGURE =====================
fig, ax = plt.subplots(1, 4, figsize=(27, 8), sharey=True)

ticksize = 30
labelsize = 30

for kk in range(4):

    cosine_similarity = []
    human_score = None

    for file in files[kk]:
        with open(file, 'rb') as f:
            data = pickle.load(f)

        cosine_similarity.append(np.array(data['Cosine similarity']))
        if human_score is None:
            human_score = np.array(data['Human score'])

    cos_by_model = {models[i]: cosine_similarity[i] for i in range(len(models))}

    bias_dots_by_model = make_shared_aggregated_dots(
        cos_by_model, human_score,
        n_dots=N_DOTS,
        subsample_n=SUB_N,
        n_bins=N_BINS,
        seed=1000 * kk
    )

    # Build dataframe
    long_rows = []
    for model in models:
        for b in bias_dots_by_model[model]:
            long_rows.append((model, b))

    df = pd.DataFrame(long_rows, columns=["Models", "Bias"])

    # Plot stripplot (uniform color)
    ax_ = sns.stripplot(
        x='Models',
        y='Bias',
        data=df,
        ax=ax[kk],
        size=5,
        color='steelblue',
        alpha=0.6,
        order=models
    )

    # Mean ± std
    means = np.array([bias_dots_by_model[m].mean() for m in models])
    stds  = np.array([bias_dots_by_model[m].std(ddof=1) for m in models])
    xpos = np.arange(len(models))

    ax[kk].errorbar(xpos, means, yerr=stds,
                    fmt='D', color='black', ecolor='black',
                    elinewidth=2, capsize=8, capthick=2,
                    markersize=10, zorder=5)

    # Paired Bonferroni tests
    pairs = list(combinations(range(len(models)), 2))
    alpha_corr = alpha / len(pairs)
    sig_pairs = []

    for i, j in pairs:
        t_stat, p = ttest_rel(
            bias_dots_by_model[models[i]],
            bias_dots_by_model[models[j]]
        )
        if p < alpha_corr:
            sig_pairs.append((i, j, p))

    sig_pairs.sort(key=lambda t: ((t[1]-t[0]), -np.log10(t[2])), reverse=True)

    top_cloud = max(np.quantile(bias_dots_by_model[m], Q_TOP) for m in models)
    y0 = top_cloud + BASE_PAD

    for idx, (i, j, p) in enumerate(sig_pairs):
        y = y0 + idx * TIER_GAP
        add_sig_bracket_onbar(ax[kk], i, j, y, BR_H, stars_from_p(p),
                              lw=BR_LW, fs=STAR_FS)

    ax[kk].axhline(0, linestyle='--', color='k', linewidth=1.5)
    ax[kk].set_title(titles[kk], fontsize=labelsize)
    ax_.set_xlabel('')
    ax_.set_xticklabels(models, fontsize=labelsize, rotation=80)
    ax_.tick_params(labelsize=ticksize)
    ax_.spines["right"].set_visible(False)
    ax_.spines["top"].set_visible(False)

ax[0].set_ylabel('Model score − Human score', fontsize=labelsize - 4)

plt.subplots_adjust(left=0.06, right=0.95, bottom=0.18, top=0.88, wspace=0.25)
plt.savefig('plots/embedding_bias_uniform_noheatmap.pdf',
            bbox_inches='tight')
plt.show()
#%%
