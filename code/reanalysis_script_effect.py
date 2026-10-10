#!/usr/bin/env python3
"""Script-versus-tier re-analysis (Section 6.1, Table 6, Appendix tables).

    python code/reanalysis_script_effect.py

From the shipped results only (no models, CPU, about 2 minutes):
  B. script effect controlling Joshi tier, Freedman-Lane permutation (20,000), per tokenizer,
     Holm-corrected, on log tokens/verse and log tokens/byte
  C. the same test on the retrieval axis
  D. variance decomposition, script vs tier
  E. planned Brahmic-vs-Latin contrast on tokens/byte
  F. script length vs tokenizer fit classification
  G. per-encoder dissociation bootstrap (own-encoder retrieval)
  H. Bosnian translator pair against the cross-language spread
Language-level analyses use n = 40 (one Bosnian translation). Outputs go to outputs/.
"""
import numpy as np, pandas as pd
from scipy import stats
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import METADATA, RESULTS, FERTILITY, outputs

RNG = np.random.default_rng(42)
OUT = str(outputs())

TOKS = ['nllb','mt5','gpt4o_o200k','mbert','xlmr','gemma','bloom','byt5','gpt4_cl100k','llama3','aya']
TOKLABEL = {'nllb':'NLLB','mt5':'mT5 / Aya-101','gpt4o_o200k':'GPT-4o (o200k)','mbert':'mBERT',
            'xlmr':'XLM-R','gemma':'Gemma-2','bloom':'BLOOM','byt5':'ByT5 (bytes)',
            'gpt4_cl100k':'GPT-4 (cl100k)','llama3':'Llama-3','aya':'Aya-101'}
ENC = {'sentence-transformers/LaBSE':'labse','BAAI/bge-m3':'bgem3',
       'intfloat/multilingual-e5-base':'e5',
       'sentence-transformers/paraphrase-multilingual-mpnet-base-v2':'mpnet'}

# ---------------------------------------------------------------- load
fert = pd.read_csv(FERTILITY)
joint = pd.read_csv(METADATA / 'joint_language_table.csv')
retr = pd.read_csv(RESULTS / 'retrieval_per_language.csv')
down = pd.read_csv(RESULTS / 'downstream_all_encoders.csv')

df = joint.merge(fert, on='language', how='left')
df['script6'] = df['script']
# Khmer script is Brahmic-derived (Pallava lineage). The paper's 6-group split
# creates a singleton group, which costs the omnibus test most of its power.
df['script5'] = df['script'].replace({'Khmer': 'Brahmic'})
# one Bosnian only for language-level inference (the pair is translator-variance evidence)
main = df[df.language != 'bosnian-mihanovich'].copy()

for t in TOKS:
    main[f'{t}_tpb'] = main[f'{t}_tok_per_verse'] / main['bytes_per_verse']

report = {}

# ---------------------------------------------------------------- helpers
def ols_r2(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot, resid, X @ beta

def dummies(labels):
    cats = sorted(set(labels))[1:]          # drop first as reference
    return np.column_stack([(np.asarray(labels) == c).astype(float) for c in cats])

def freedman_lane(y, script_labels, tier, n_perm=20000, rng=RNG):
    """Permutation test for the script block controlling resource tier (Freedman-Lane)."""
    tier = np.asarray(tier, float).reshape(-1, 1)
    S = dummies(script_labels)
    r2_red, resid_red, fit_red = ols_r2(y, tier)
    r2_full, _, _ = ols_r2(y, np.column_stack([tier, S]))
    d_obs = r2_full - r2_red
    null = np.empty(n_perm)
    n = len(y)
    for i in range(n_perm):
        ystar = fit_red + resid_red[rng.permutation(n)]
        r2f, _, _ = ols_r2(ystar, np.column_stack([tier, S]))
        r2r, _, _ = ols_r2(ystar, tier)
        null[i] = r2f - r2r
    p = (1 + (null >= d_obs).sum()) / (1 + n_perm)
    return d_obs, p, r2_red, r2_full

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        run = max(run, val); adj[idx] = min(run, 1.0)
    return adj

def spearman_partial(x, y, z):
    """Rank partial correlation of x,y controlling z."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a, b):
        b1 = np.column_stack([np.ones(len(b)), b])
        beta, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ beta
    return stats.pearsonr(resid(rx, rz), resid(ry, rz))[0]

# =================================================================
# A. Reproduce the paper's omnibus test, then re-specify it
# =================================================================
y_gpt4o = main['gpt4o_o200k_tok_per_verse'].values
g6 = [main[main.script6 == s]['gpt4o_o200k_tok_per_verse'].values for s in sorted(set(main.script6))]
H6, p6 = stats.kruskal(*g6)
g5 = [main[main.script5 == s]['gpt4o_o200k_tok_per_verse'].values for s in sorted(set(main.script5))]
H5, p5 = stats.kruskal(*g5)
report['A_omnibus'] = {
    'paper_6group': {'H': round(H6, 2), 'p': round(p6, 4), 'k': len(g6)},
    'regrouped_5group': {'H': round(H5, 2), 'p': round(p5, 4), 'k': len(g5)},
}

# =================================================================
# B. Script effect controlling resource tier, all 11 tokenizers
#    outcome 1: log tokens/verse (what users are billed)
#    outcome 2: log tokens/byte  (tokenizer fit, script length removed)
# =================================================================
rows = []
for t in TOKS:
    for outcome, col in [('tokens/verse', f'{t}_tok_per_verse'), ('tokens/byte', f'{t}_tpb')]:
        y = np.log(main[col].values)
        d, p, r2r, r2f = freedman_lane(y, main['script5'].values, main['joshi'].values)
        rows.append({'tokenizer': TOKLABEL[t], 'outcome': outcome,
                     'r2_tier': round(r2r, 3), 'r2_tier_script': round(r2f, 3),
                     'delta_r2': round(d, 3), 'p_perm': p})
scr = pd.DataFrame(rows)
for outcome in ['tokens/verse', 'tokens/byte']:
    m = scr.outcome == outcome
    scr.loc[m, 'p_holm'] = holm(scr.loc[m, 'p_perm'].values)
scr['p_perm'] = scr['p_perm'].round(4); scr['p_holm'] = scr['p_holm'].round(4)
scr.to_csv(f'{OUT}/script_effect_by_tokenizer.csv', index=False)
report['B_script_effect'] = {
    'n_sig_holm_tokens_per_verse': int((scr[(scr.outcome=='tokens/verse')].p_holm < 0.05).sum()),
    'n_sig_holm_tokens_per_byte': int((scr[(scr.outcome=='tokens/byte')].p_holm < 0.05).sum()),
    'n_tokenizers': len(TOKS),
}

# =================================================================
# C. The same test on the retrieval axis -> the dissociation
# =================================================================
ret_wide = retr.assign(enc=retr.model.map(ENC)).pivot(index='language', columns='enc', values='mean_top1')
main = main.merge(ret_wide, left_on='language', right_index=True, how='left', suffixes=('', '_e'))
ret_rows = []
for enc in ['labse', 'bgem3', 'e5', 'mpnet']:
    sub = main.dropna(subset=[enc])
    y = sub[enc].values
    d, p, r2r, r2f = freedman_lane(y, sub['script5'].values, sub['joshi'].values)
    ret_rows.append({'encoder': enc, 'r2_tier': round(r2r, 3), 'r2_tier_script': round(r2f, 3),
                     'delta_r2': round(d, 3), 'p_perm': round(p, 4)})
# pooled retrieval (paper's 'retrieval' column = mean over encoders)
d, p, r2r, r2f = freedman_lane(main['retrieval'].values, main['script5'].values, main['joshi'].values)
ret_rows.append({'encoder': 'pooled (mean)', 'r2_tier': round(r2r, 3), 'r2_tier_script': round(r2f, 3),
                 'delta_r2': round(d, 3), 'p_perm': round(p, 4)})
ret_scr = pd.DataFrame(ret_rows)
ret_scr.to_csv(f'{OUT}/script_effect_retrieval.csv', index=False)

# =================================================================
# D. Variance decomposition: does script explain fertility and tier explain retrieval?
# =================================================================
dec = []
S = dummies(main['script5'].values); T = main['joshi'].values.reshape(-1, 1)
for name, y in [('fertility (GPT-4o, log)', np.log(main['gpt4o_o200k_tok_per_verse'].values)),
                ('fertility (mean of 11, log)',
                 np.log(main[[f'{t}_tok_per_verse' for t in TOKS]].mean(axis=1).values)),
                ('tokenizer fit (GPT-4o tok/byte, log)', np.log(main['gpt4o_o200k_tpb'].values)),
                ('retrieval (pooled)', main['retrieval'].values)]:
    r2_s, _, _ = ols_r2(y, S)
    r2_t, _, _ = ols_r2(y, T)
    r2_b, _, _ = ols_r2(y, np.column_stack([T, S]))
    dec.append({'outcome': name, 'R2_script_only': round(r2_s, 3), 'R2_tier_only': round(r2_t, 3),
                'R2_both': round(r2_b, 3), 'script_unique': round(r2_b - r2_t, 3),
                'tier_unique': round(r2_b - r2_s, 3)})
decomp = pd.DataFrame(dec)
decomp.to_csv(f'{OUT}/variance_decomposition.csv', index=False)

# =================================================================
# E. Planned contrast: Brahmic-derived vs Latin (one df, full power)
# =================================================================
contr = []
for t in TOKS:
    a = np.log(main[main.script5 == 'Brahmic'][f'{t}_tpb'].values)
    b = np.log(main[main.script5 == 'Latin'][f'{t}_tpb'].values)
    U, pu = stats.mannwhitneyu(a, b, alternative='two-sided')
    # rank-biserial effect size
    rb = 1 - 2 * U / (len(a) * len(b))
    contr.append({'tokenizer': TOKLABEL[t], 'brahmic_median_tpb': round(np.exp(np.median(a)), 4),
                  'latin_median_tpb': round(np.exp(np.median(b)), 4),
                  'ratio': round(np.exp(np.median(a) - np.median(b)), 2),
                  'U': U, 'p': round(pu, 5), 'rank_biserial': round(rb, 3)})
cdf = pd.DataFrame(contr); cdf['p_holm'] = holm(cdf.p.values).round(5)
cdf.to_csv(f'{OUT}/contrast_brahmic_vs_latin.csv', index=False)
report['E_contrast'] = {'n_sig_holm': int((cdf.p_holm < 0.05).sum()), 'n': len(cdf)}

# =================================================================
# F. Quantitative long/short script + tokenizer fit
# =================================================================
en_bytes = float(main[main.language == 'english']['bytes_per_verse'].iloc[0])
cls = main[['language', 'script5', 'joshi', 'bytes_per_verse']].copy()
cls['bytes_vs_en'] = (cls.bytes_per_verse / en_bytes).round(2)
cls['gpt4o_tpv'] = main['gpt4o_o200k_tok_per_verse'].round(1)
cls['gpt4o_tpb'] = main['gpt4o_o200k_tpb'].round(3)
en_tpb = float(main[main.language == 'english']['gpt4o_o200k_tpb'].iloc[0])
cls['fit_vs_en'] = (cls.gpt4o_tpb / en_tpb).round(2)
# explicit, pre-registered thresholds
cls['script_length'] = np.where(cls.bytes_vs_en >= 1.5, 'long', np.where(cls.bytes_vs_en <= 1.15, 'short', 'mid'))
cls['tokenizer_fit'] = np.where(cls.fit_vs_en >= 1.5, 'poor', np.where(cls.fit_vs_en <= 1.15, 'good', 'mid'))
cls = cls.sort_values('bytes_vs_en', ascending=False)
cls.to_csv(f'{OUT}/script_length_vs_fit.csv', index=False)
report['F_2x2'] = cls.groupby(['script_length', 'tokenizer_fit']).size().to_dict()
report['F_2x2'] = {f'{k[0]}|{k[1]}': int(v) for k, v in report['F_2x2'].items()}
# the Khmer / Chinese comparison discussed in Section 6.1
kh = cls[cls.language == 'khmer'].iloc[0]; ch = cls[cls.language == 'chinese'].iloc[0]
report['F_khmer_vs_chinese'] = {
    'khmer': {'bytes_vs_en': float(kh.bytes_vs_en), 'fit_vs_en': float(kh.fit_vs_en)},
    'chinese': {'bytes_vs_en': float(ch.bytes_vs_en), 'fit_vs_en': float(ch.fit_vs_en)}}

# =================================================================
# G. How strong is the dissociation? Bootstrap CI + equivalence test
#    A non-significant partial is a failure to reject, so bound it with an interval.
# =================================================================
down['enc'] = down['encoder']
diss = []
for enc in ['labse', 'bgem3', 'e5', 'mpnet']:
    d_enc = down[down.enc == enc][['language', 'macro_f1']]
    sub = main.merge(d_enc, on='language', how='inner').dropna(subset=[enc, 'macro_f1'])
    transfer = sub['macro_f1'].values
    retrieval = sub[enc].values
    fertility = sub['gpt4o_o200k_tok_per_verse'].values
    r_ret = stats.spearmanr(transfer, retrieval)[0]
    r_fert = stats.spearmanr(transfer, fertility)[0]
    pr = spearman_partial(transfer, fertility, retrieval)
    boot = []
    n = len(sub)
    for _ in range(10000):
        idx = RNG.integers(0, n, n)
        if len(set(idx)) < 5: continue
        try: boot.append(spearman_partial(transfer[idx], fertility[idx], retrieval[idx]))
        except Exception: pass
    lo, hi = np.percentile(boot, [2.5, 97.5])
    diss.append({'encoder': enc, 'n_lang': n, 'rho_retrieval': round(r_ret, 3),
                 'rho_fertility': round(r_fert, 3), 'partial_fert_given_ret': round(pr, 3),
                 'boot_ci_lo': round(lo, 3), 'boot_ci_hi': round(hi, 3),
                 'within_pm0.3': bool(abs(lo) < 0.3 and abs(hi) < 0.3)})
ddf = pd.DataFrame(diss)
ddf.to_csv(f'{OUT}/dissociation_bootstrap.csv', index=False)
report['G_dissociation'] = {'n_encoders_bounded_within_0.3': int(ddf['within_pm0.3'].sum())}

# =================================================================
# H. Translator variance, restated against the cross-language spread
# =================================================================
bos = df[df.language.str.startswith('bosnian')]
tv = {}
for t in TOKS:
    v = bos[f'{t}_tok_per_verse'].values
    tv[TOKLABEL[t]] = round(float(max(v) / min(v)), 4)
cross = {}
for t in TOKS:
    v = main[f'{t}_tok_per_verse'].values
    cross[TOKLABEL[t]] = round(float(np.percentile(v, 90) / np.percentile(v, 10)), 3)
tvdf = pd.DataFrame({'tokenizer': list(tv), 'bosnian_pair_ratio': list(tv.values()),
                     'cross_language_p90_p10': [cross[k] for k in tv]})
tvdf['ratio_of_ratios'] = (((tvdf.cross_language_p90_p10 - 1) /
                            (tvdf.bosnian_pair_ratio - 1).replace(0, np.nan))).round(1)
tvdf.to_csv(f'{OUT}/translator_variance_bosnian.csv', index=False)

with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
print('\n--- B: script effect controlling tier ---')
print(scr.to_string(index=False))
print('\n--- C: retrieval axis ---')
print(ret_scr.to_string(index=False))
print('\n--- D: variance decomposition ---')
print(decomp.to_string(index=False))
print('\n--- E: Brahmic vs Latin (tokens/byte) ---')
print(cdf.to_string(index=False))
print('\n--- G: dissociation ---')
print(ddf.to_string(index=False))
print('\n--- H: translator vs cross-language ---')
print(tvdf.to_string(index=False))
