# Two Axes of Multilingual Inequity: Corpus, Code, and Results

Anonymous release accompanying the EACL 2027 submission
"Two Axes of Multilingual Inequity: Tokenizer Fairness and Retrieval Coverage on a
Meaning-Controlled 41-Language Corpus".

## What this is

A meaning-controlled multi-parallel corpus built from the Qur'an: 6,236 verses translated into
40 languages by professional translators within one publishing project (QuranEnc / IslamHouse),
verse-aligned with zero missing cells (41 translation files; Bosnian appears in two independent
translations, which serve as a translator-variation control). The Arabic source text (Tanzil
Uthmani) is added as the source-language baseline: 41 languages, 42 files in total.

On this corpus the paper measures: (1) tokenizer fertility across eleven production tokenizers,
(2) cross-lingual verse retrieval across four multilingual sentence encoders, (3) a zero-shot
Meccan/Medinan downstream transfer probe, plus a FLORES-200 replication and four quantified
robustness checks.

## Layout

```
data/
  translations.zip              41 translation files (CSV: id, sura, aya, translation, footnotes)
  metadata/
    joint_language_table.csv    per-language script, Joshi (2020) resource tier (verified against
                                the released lang2tax taxonomy), GPT-4o fertility, retrieval mean
    sura_labels.csv             Meccan/Medinan label per sura (86/28, standard division)
    duplicate_groups_english.csv  91 near-duplicate verse groups (token Jaccard >= 0.8, English),
                                272 member verses, for duplicate-aware retrieval scoring
code/
  01_tokenizer_fertility.ipynb  Experiment 1: fertility, 11 tokenizers (Colab)
  02_crosslingual_retrieval.ipynb  Experiment 2: retrieval, 4 encoders (Colab)
  03_stats_and_paper.ipynb      bootstrap CIs, correlations, tables
  04_flores_replication_and_downstream.ipynb  FLORES-200 replication + downstream probe (Colab)
  flores_fertility.py           FLORES-200 fertility replication (script version)
  flores_retrieval.py           FLORES-200 retrieval-floor check for low-resource languages
  downstream_transfer.py        Meccan/Medinan zero-shot transfer probe (script version)
  robustness_checks.py          near-duplicates, truncation, translator verbosity, length probe,
                                tokenizer-mirror verification
  get_arabic_source.py          downloads the Tanzil Uthmani Arabic source text
results/
  fertility_production_v2_all11.csv   tokens/verse, 11 tokenizers x 42 files, + ratios
  fertility_tokenizer_ci.csv          bootstrap CIs, p90/p10 and spread per tokenizer
  fertility_controlled_production.csv matched-size English-only vs multilingual 32k BPE contrast
  cost_simulation.csv                 dollar and context-window conversions
  retrieval_pairs.csv                 1,640 ordered language pairs x 4 encoders (top-1, MRR)
  retrieval_per_language.csv          per-language retrieval means
  retrieval_encoder_ci.csv, retrieval_language_ci.csv  bootstrap CIs
  flores_fertility_by_language.csv    FLORES-200 devtest fertility per tokenizer/language
  flores_vs_corpus_summary.csv        corpus-vs-FLORES inequality comparison (rank rho = 0.92)
  flores_retrieval_floor.csv          FLORES-200 English-paired retrieval for floor languages
  downstream_all_encoders.csv         Meccan/Medinan transfer, 41 languages x 4 encoders
  truncation_rates.csv                % verses over each encoder's sequence limit, per language
  length_probe.csv                    length-only probe baseline per language
```

## Reproducing

Python 3.10+. `pip install -r requirements.txt`. The notebooks are Colab-ready (GPU recommended
for experiments 2 and 4; everything else runs on CPU). FLORES-200 devtest downloads from the
official Meta URL inside the scripts. The retrieval sample, probe splits, and bootstrap seeds are
fixed (seed 42) and identical across languages and encoders.

Tokenizer checkpoints: tiktoken `o200k_base` / `cl100k_base`; HF `bert-base-multilingual-cased`,
`xlm-roberta-base`, `bigscience/bloom-560m`, `facebook/nllb-200-distilled-600M`,
`google/mt5-small` (also the Aya-101 open-release vocabulary), `google/byt5-small`;
Llama-3 and Gemma-2 tokenizers via ungated mirrors (`unsloth/llama-3-8b`, `unsloth/gemma-2-2b`),
verified to produce token counts identical to two further independent mirrors each on a
multilingual sample (`robustness_checks.py`).

Encoders: `sentence-transformers/LaBSE` (limit 256),
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (128),
`intfloat/multilingual-e5-base` (512, "query:" prefix), `BAAI/bge-m3` (8192).

## Licensing

- **Translations** (`data/translations.zip`): distributed by IslamHouse.com under the MIT
  License; see `DATA_LICENSE.md` for the notice. Redistributed here unmodified, with per-file
  translation identifiers and provenance headers intact.
- **Arabic source**: not bundled; `code/get_arabic_source.py` fetches the Tanzil Uthmani text,
  whose terms permit redistribution of the unmodified text with attribution (https://tanzil.net).
- **Code and derived result files**: MIT License (`LICENSE`).

## Notes for reviewers

The corpus supports evaluation and controlled analysis. Known residual limitations are quantified
in the paper's Robustness section: near-duplicate verses (about 1.7% of a 1,000-verse sample has
an in-sample near-duplicate; multilingual spot-check rates 2.0-5.2%), MPNet truncation (up to 20%
in high-fertility languages; the untruncated encoders carry the conclusions), translator
verbosity (Bosnian pair differs by 0.4% in length), and the length component of the probe
(macro-F1 0.601, uncorrelated with the retrieval axis).
