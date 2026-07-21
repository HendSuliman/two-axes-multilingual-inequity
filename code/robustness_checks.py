#!/usr/bin/env python3
"""Robustness checks for the two-axes paper (Section 'Robustness checks').

1. Near-duplicate verse quantification (English word-level Jaccard; character
   4-gram Jaccard spot checks for Khmer/Japanese; word-level for Yoruba/Malayalam),
   plus the expected in-sample duplicate rate for a random 1,000-verse sample.
2. Per-encoder truncation exposure: % of verses over each encoder's sequence limit.
3. Translator-verbosity control: the two independent Bosnian translations.
4. Length-only probe baseline for the Meccan/Medinan task.
5. Tokenizer-mirror verification: unsloth mirrors vs independent mirrors.

Expects the unzipped corpus in ./corpus and metadata CSVs alongside.
Run: python robustness_checks.py
"""
import io, re, itertools
import numpy as np, pandas as pd
from pathlib import Path

SEED = 42
MEDINAN = {2,3,4,5,8,9,13,22,24,33,47,48,49,55,57,58,59,60,61,62,63,64,65,66,76,98,99,110}

def read_corpus_csv(path):
    txt = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    m = re.search(r"^id,sura,aya", txt, flags=re.M)
    df = pd.read_csv(io.StringIO(txt[m.start():]))[["sura","aya","translation"]].dropna()
    df["sura"] = df.sura.astype(int); df["aya"] = df.aya.astype(int)
    return df

def jaccard_near_dup_pairs(texts, mode="word", thresh=0.8):
    if mode == "word":
        units = [set(re.sub(r"[^\w ]", "", str(t).lower()).split()) for t in texts]
    else:
        def grams(t):
            t = re.sub(r"\s+", "", str(t).lower())
            return set(t[i:i+4] for i in range(max(1, len(t)-3)))
        units = [grams(t) for t in texts]
    n = len(units); order = sorted(range(n), key=lambda i: len(units[i])); pairs = []
    for ii in range(n):
        i = order[ii]; li = len(units[i])
        if li == 0: continue
        for jj in range(ii+1, n):
            j = order[jj]; lj = len(units[j])
            if lj > li/thresh + 1: break
            inter = len(units[i] & units[j])
            if inter / max(1, (li + lj - inter)) >= thresh: pairs.append((i, j))
    return pairs

def duplicates():
    en = read_corpus_csv(next(Path("corpus").glob("english_*.csv")))
    pairs = jaccard_near_dup_pairs(en.translation.tolist(), "word")
    near = set(i for p in pairs for i in p)
    print(f"EN: {len(near)}/{len(en)} verses with a near-duplicate ({len(near)/len(en):.1%}), {len(pairs)} pairs")
    rng = np.random.default_rng(0); rates = []
    for _ in range(500):
        s = set(rng.choice(len(en), 1000, replace=False))
        affected = set(i for i, j in pairs if i in s and j in s for i in (i, j))
        rates.append(len(affected)/1000)
    print(f"in-sample rate for a 1000-verse sample: mean={np.mean(rates):.2%}, p95={np.percentile(rates,95):.2%}")
    for pat, mode in [("khmer_*", "char4"), ("japanese_*", "char4"),
                      ("yoruba_*", "word"), ("malayalam_*", "word")]:
        f = next(Path("corpus").glob(pat + ".csv"), None) or next(Path("corpus").glob(pat), None)
        f = f or sorted(Path("corpus").glob(pat.split("_")[0] + "*"))[0]
        texts = read_corpus_csv(f).translation.tolist()
        p = jaccard_near_dup_pairs(texts, mode)
        near = set(i for q in p for i in q)
        print(f"{f.name.split('_')[0]} ({mode}): {len(near)/len(texts):.1%}")

def truncation():
    from transformers import AutoTokenizer
    ENC = {"labse": ("sentence-transformers/LaBSE", 256),
           "mpnet": ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 128),
           "e5": ("intfloat/multilingual-e5-base", 512),
           "bgem3": ("BAAI/bge-m3", 8192)}
    toks = {k: AutoTokenizer.from_pretrained(name) for k, (name, cap) in ENC.items()}
    rows = []
    for f in sorted(Path("corpus").glob("*.csv")):
        texts = read_corpus_csv(f).translation.astype(str).tolist()
        for k, (name, cap) in ENC.items():
            lens = np.array([len(x) for x in toks[k](texts, add_special_tokens=True, truncation=False)["input_ids"]])
            rows.append({"file": f.name, "encoder": k, "pct_over_cap": float((lens > cap).mean()*100)})
    df = pd.DataFrame(rows); df.to_csv("truncation_rates.csv", index=False)
    print(df.groupby("encoder").pct_over_cap.agg(["mean", "max"]).round(3))

def bosnian_verbosity():
    import tiktoken
    o200k = tiktoken.get_encoding("o200k_base")
    files = sorted(Path("corpus").glob("bosnian_*.csv"))
    stats = {}
    for f in files:
        df = read_corpus_csv(f)
        b = df.translation.astype(str).str.encode("utf-8").str.len()
        stats[f.name] = b
        t = np.mean([len(o200k.encode(s, disallowed_special=())) for s in df.translation.astype(str)])
        print(f"{f.name}: bytes/verse={b.mean():.1f}, o200k tok/verse={t:.2f}")
    a, b = stats.values()
    print(f"byte ratio {max(a.mean(),b.mean())/min(a.mean(),b.mean()):.3f}, per-verse length corr r={np.corrcoef(a,b)[0,1]:.3f}")

def length_probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(SEED)
    en = read_corpus_csv(next(Path("corpus").glob("english_*.csv")))
    keys = list(en[["sura","aya"]].itertuples(index=False, name=None))
    idx = rng.permutation(len(keys))
    eval_keys = [keys[i] for i in sorted(idx[:1000])]
    train_keys = [keys[i] for i in sorted(idx[1000:3000])]
    lab = lambda ks: np.array([1 if s in MEDINAN else 0 for s, a in ks])
    def feats(df, ks):
        d = df.set_index(["sura","aya"])["translation"]
        ts = [str(d.get(k, "")) for k in ks]
        return np.c_[np.log1p([len(t.encode("utf-8")) for t in ts]),
                     np.log1p([len(t.split()) for t in ts])]
    Xtr = feats(en, train_keys); mu, sd = Xtr.mean(0), Xtr.std(0)
    clf = LogisticRegression(max_iter=2000).fit((Xtr-mu)/sd, lab(train_keys))
    rows = []
    for f in sorted(Path("corpus").glob("*.csv")):
        X = feats(read_corpus_csv(f), eval_keys)
        rows.append({"file": f.name,
                     "macro_f1": f1_score(lab(eval_keys), clf.predict((X-X.mean(0))/X.std(0)), average="macro")})
    df = pd.DataFrame(rows); df.to_csv("length_probe.csv", index=False)
    print(f"length-only probe: mean={df.macro_f1.mean():.3f}, sd={df.macro_f1.std():.3f}")

def tokenizer_mirrors():
    from transformers import AutoTokenizer
    sample_files = ["english", "khmer", "malayalam", "yoruba", "japanese"]
    samples = []
    for s in sample_files:
        samples += read_corpus_csv(sorted(Path("corpus").glob(s + "*"))[0]).translation.astype(str).head(50).tolist()
    for family, mirrors in [("llama3", ["unsloth/llama-3-8b", "Xenova/llama-3-tokenizer", "NousResearch/Meta-Llama-3-8B"]),
                            ("gemma2", ["unsloth/gemma-2-2b", "Xenova/gemma-2-tokenizer", "unsloth/gemma-2-9b"])]:
        counts = {}
        for m in mirrors:
            t = AutoTokenizer.from_pretrained(m)
            counts[m] = [len(t.encode(s, add_special_tokens=False)) for s in samples]
        for a, b in itertools.combinations(counts, 2):
            same = sum(x == y for x, y in zip(counts[a], counts[b]))
            print(f"{family}: {a} vs {b}: {same}/{len(samples)} identical")

if __name__ == "__main__":
    duplicates(); truncation(); bosnian_verbosity(); length_probe(); tokenizer_mirrors()
