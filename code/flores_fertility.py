#!/usr/bin/env python3
"""FLORES-200 devtest fertility replication of the Qur'an-corpus tokenizer audit.

For each corpus language mapped to a FLORES-200 code, compute mean tokens per
sentence on the 1,012 devtest sentences under the same production tokenizers,
then compare per-tokenizer inequality (p90/p10, max/min) and per-language
fertility (Spearman) against the corpus numbers.
"""
import json, numpy as np, pandas as pd
from pathlib import Path

FLORES_DIR = Path("flores200_dataset/devtest")
RNG = np.random.default_rng(42)

# corpus-language -> FLORES-200 code (Kurdish translation is Sorani, Arabic script)
LANG2FLORES = {
    "albanian": "als_Latn", "asante": "twi_Latn", "assamese": "asm_Beng",
    "azeri": "azj_Latn", "bosnian-mihanovich": "bos_Latn", "bosnian-rwwad": "bos_Latn",
    "chinese": "zho_Hans", "croatian": "hrv_Latn", "dutch": "nld_Latn",
    "english": "eng_Latn", "french": "fra_Latn", "german": "deu_Latn",
    "gujarati": "guj_Gujr", "hausa": "hau_Latn", "indonesian": "ind_Latn",
    "japanese": "jpn_Jpan", "kannada": "kan_Knda", "khmer": "khm_Khmr",
    "kurdish": "ckb_Arab", "kyrgyz": "kir_Cyrl", "lingala": "lin_Latn",
    "lithuanian": "lit_Latn", "macedonian": "mkd_Cyrl", "malayalam": "mal_Mlym",
    "moore": "mos_Latn", "oromo": "gaz_Latn", "pashto": "pbt_Arab",
    "persian": "pes_Arab", "portuguese": "por_Latn", "romanian": "ron_Latn",
    "serbian": "srp_Cyrl", "spanish": "spa_Latn", "tagalog": "tgl_Latn",
    "tajik": "tgk_Cyrl", "tamil": "tam_Taml", "turkish": "tur_Latn",
    "urdu": "urd_Arab", "uyghur": "uig_Arab", "uzbek": "uzn_Latn",
    "vietnamese": "vie_Latn", "yoruba": "yor_Latn", "arabic_source": "arb_Arab",
}

CORPUS_COL = {  # tokenizer key -> column in fertility_production_v2_all11.csv
    "gpt4o_o200k": "gpt4o_o200k_tok_per_verse", "gpt4_cl100k": "gpt4_cl100k_tok_per_verse",
    "llama3": "llama3_tok_per_verse", "gemma2": "gemma_tok_per_verse",
    "mbert": "mbert_tok_per_verse", "xlmr": "xlmr_tok_per_verse",
    "bloom": "bloom_tok_per_verse", "nllb": "nllb_tok_per_verse",
    "mt5": "mt5_tok_per_verse", "byt5_bytes": "byt5_tok_per_verse",
}

def get_tokenizers():
    import tiktoken
    from transformers import AutoTokenizer
    toks = {}
    o200k = tiktoken.get_encoding("o200k_base"); cl100k = tiktoken.get_encoding("cl100k_base")
    toks["gpt4o_o200k"] = lambda s: len(o200k.encode(s, disallowed_special=()))
    toks["gpt4_cl100k"] = lambda s: len(cl100k.encode(s, disallowed_special=()))
    hf = {
        "mbert": ["bert-base-multilingual-cased"],
        "xlmr": ["xlm-roberta-base", "FacebookAI/xlm-roberta-base"],
        "bloom": ["bigscience/bloom-560m", "bigscience/tokenizer"],
        "nllb": ["facebook/nllb-200-distilled-600M"],
        "mt5": ["google/mt5-small"],
        "llama3": ["unsloth/llama-3-8b", "Xenova/llama-3-tokenizer", "unsloth/llama-3-8b-Instruct"],
        "gemma2": ["unsloth/gemma-2-2b", "unsloth/gemma-2-9b"],
    }
    for key, names in hf.items():
        for n in names:
            try:
                t = AutoTokenizer.from_pretrained(n)
                toks[key] = (lambda tt: (lambda s: len(tt.encode(s, add_special_tokens=False))))(t)
                print(f"loaded {key} <- {n}")
                break
            except Exception as e:
                print(f"  {key}: {n} failed ({type(e).__name__})")
    toks["byt5_bytes"] = lambda s: len(s.encode("utf-8"))
    return toks

def ineq(v):
    v = np.asarray(v, float)
    return np.percentile(v, 90) / np.percentile(v, 10), v.max() / v.min()

def boot_p90p10(v, n=10000):
    v = np.asarray(v, float); out = []
    for _ in range(n):
        b = RNG.choice(v, size=len(v), replace=True)
        out.append(np.percentile(b, 90) / np.percentile(b, 10))
    return np.percentile(out, [2.5, 97.5])

def main():
    from scipy.stats import spearmanr
    toks = get_tokenizers()
    print("tokenizers ready:", sorted(toks))
    # load FLORES sentences for the unique codes we need
    codes = sorted(set(LANG2FLORES.values()))
    sents = {c: (FLORES_DIR / f"{c}.devtest").read_text(encoding="utf-8").rstrip("\n").split("\n") for c in codes}
    for c in codes:
        assert len(sents[c]) == 1012, (c, len(sents[c]))
    # mean tokens/sentence per (tokenizer, code)
    rows = []
    for tk, fn in toks.items():
        for c in codes:
            counts = [fn(s) for s in sents[c]]
            rows.append({"tokenizer": tk, "flores_code": c,
                         "mean_tok_per_sent": float(np.mean(counts))})
        print("done", tk)
    fl = pd.DataFrame(rows)
    fl.to_csv("flores_fertility_by_language.csv", index=False)

    corpus = pd.read_csv("/mnt/user-data/uploads/Quran/fertility_production_v2_all11.csv")
    corpus = corpus[corpus.language.isin(LANG2FLORES)].copy()
    corpus["flores_code"] = corpus.language.map(LANG2FLORES)
    # drop duplicate bosnian (two translations -> same FLORES code): average them
    summary = []
    for tk, col in CORPUS_COL.items():
        if tk not in toks: continue
        cg = corpus.groupby("flores_code")[col].mean()
        fg = fl[fl.tokenizer == tk].set_index("flores_code")["mean_tok_per_sent"]
        common = sorted(set(cg.index) & set(fg.index))
        rho, p = spearmanr(cg[common], fg[common])
        qp, qs = ineq(cg[common]); fp, fs = ineq(fg[common])
        lo, hi = boot_p90p10(fg[common].values)
        summary.append({"tokenizer": tk, "n_langs": len(common),
                        "corpus_p90p10": round(qp, 2), "flores_p90p10": round(fp, 2),
                        "flores_p90p10_lo": round(lo, 2), "flores_p90p10_hi": round(hi, 2),
                        "corpus_maxmin": round(qs, 2), "flores_maxmin": round(fs, 2),
                        "spearman_lang_fertility": round(rho, 3), "p": f"{p:.1e}"})
    sm = pd.DataFrame(summary).sort_values("flores_p90p10")
    sm.to_csv("flores_vs_corpus_summary.csv", index=False)
    print(sm.to_string(index=False))
    # rank agreement of tokenizer inequality between the two corpora
    rho_rank, p_rank = spearmanr(sm.corpus_p90p10, sm.flores_p90p10)
    print(f"\nTokenizer inequality rank agreement (p90/p10): rho={rho_rank:.3f} p={p_rank:.2e}")
    json.dump({"rho_rank": rho_rank, "p_rank": p_rank}, open("flores_rank_agreement.json", "w"))

if __name__ == "__main__":
    main()
