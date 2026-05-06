import re
import pandas as pd
from BPE import BPE

df = pd.read_csv("VAIO_CNJ.csv")
df = df[df["Form1Surface"].notna() & df["Form1Split"].notna()].copy()

all_surfaces = []
for _, row in df.iterrows():
    for i in [1, 2, 3]:
        s = row.get(f"Form{i}Surface")
        if pd.notna(s) and str(s).strip() != "":
            all_surfaces.append(str(s))

VOCAB_SIZE = 200

bpe = BPE(corpus=all_surfaces, vocab_size=VOCAB_SIZE)
bpe.train()


def get_segmentation(form_split):
    match = re.match(r'<<(.+?)>>(.*)', str(form_split))
    if not match:
        return None, None
    stem   = match.group(1)
    suffix = match.group(2)
    if suffix:
        return stem + " + " + suffix, suffix
    return stem, ""


def get_verdict(bpe_result, correct_result):
    bpe_pieces = bpe_result.split(" | ")
    if len(bpe_pieces) == 1:
        return "Erase"
    if bpe_result == correct_result.replace(" + ", " | "):
        return "Preserve"
    return "Distort"


results = []

for _, row in df.iterrows():
    forms = []
    for i in [1, 2, 3]:
        surface    = row.get(f"Form{i}Surface")
        form_split = row.get(f"Form{i}Split")
        if pd.notna(surface) and str(surface).strip() != "":
            forms.append((str(surface), str(form_split)))

    for surface, form_split in forms:
        bpe_tokens     = bpe.tokenize(surface)
        bpe_result     = " | ".join(bpe_tokens)
        correct_result, gold_suffix = get_segmentation(form_split)
        if correct_result is None:
            continue
        is_correct = (bpe_result == correct_result.replace(" + ", " | "))
        verdict    = get_verdict(bpe_result, correct_result)

        results.append({
            "word":           surface,
            "bpe_result":     bpe_result,
            "correct_result": correct_result,
            "gold_suffix":    gold_suffix,
            "is_correct":     is_correct,
            "verdict":        verdict,
            "negation":       row["Negation"],
            "mode":           row["Mode"],
        })

out = pd.DataFrame(results)
out.to_csv("bpe_experiment_results_200.csv", index=False)

total   = len(out)
correct = out["is_correct"].sum()
print(f"Total words : {total}")
print(f"Correct     : {correct} ({100*correct/total:.1f}%)")
print(f"Wrong       : {total - correct} ({100*(total-correct)/total:.1f}%)")