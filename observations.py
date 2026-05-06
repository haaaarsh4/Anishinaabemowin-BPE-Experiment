import pandas as pd

v50  = pd.read_csv("bpe_experiment_results_50.csv")
v100 = pd.read_csv("bpe_experiment_results_100.csv")
v200 = pd.read_csv("bpe_experiment_results_200.csv")

for df in [v50, v100, v200]:
    df["suffix_len"] = df["gold_suffix"].str.len()
    df["word_len"] = df["word"].str.len()

def pcts(df, mask=None):
    subset = df[mask] if mask is not None else df
    total  = len(subset)
    return {
        "Preserve": round(100 * (subset["verdict"] == "Preserve").sum() / total, 1),
        "Distort":  round(100 * (subset["verdict"] == "Distort").sum()  / total, 1),
        "Erase":    round(100 * (subset["verdict"] == "Erase").sum()    / total, 1),
    }

rows = []

for label, df in [("vocab_50", v50), ("vocab_100", v100), ("vocab_200", v200)]:
    r = pcts(df); r["condition"] = label; rows.append(r)

for neg in ["Neg", "Pos"]:
    r = pcts(v50, v50["negation"] == neg); r["condition"] = f"negation_{neg}"; rows.append(r)

for mode in v50["mode"].unique():
    r = pcts(v50, v50["mode"] == mode); r["condition"] = f"mode_{mode}"; rows.append(r)

for label, mask in [("short_suffix", v50["suffix_len"] <= 6), ("long_suffix", v50["suffix_len"] > 6)]:
    r = pcts(v50, mask); r["condition"] = label; rows.append(r)

for label, mask in [("short_word", v50["word_len"] <= 12), ("long_word", v50["word_len"] > 12)]:
    r = pcts(v50, mask); r["condition"] = label; rows.append(r)

out = pd.DataFrame(rows)[["condition", "Preserve", "Distort", "Erase"]]
out.to_csv("observations.csv", index=False)
print("Saved to observations.csv")