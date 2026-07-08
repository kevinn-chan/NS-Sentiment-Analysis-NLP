"""
Trial commitment_v2_prompt on blind_balanced_100.csv (has human labels).
Compares new prompt output vs human labels.
Prints κ, F1, confusion matrix, and flags disagreements for review.

Usage: python scripts/trial_prompt_v2.py
"""
import json, os, sys, time
import pandas as pd
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report, confusion_matrix
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, "src")
from features.prompts.commitment_v2_prompt import SYSTEM_PROMPT, build_user_message

CSV  = "data/processed/new/blind_balanced_100.csv"
OUT  = "data/processed/new/trial_v2_results.csv"
MODEL = "gpt-4.1"   # use full 4.1 for quality trial

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

df = pd.read_csv(CSV)
df = df[df["human_label"].str.strip() != ""].copy()
print(f"Trialling on {len(df)} annotated rows from blind_balanced_100.csv\n")

def label_row(text: str) -> dict:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_message(text)},
                ],
                max_tokens=60,
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  retry {attempt+1}: {e}")
            time.sleep(2)
    return {"buyin": "neutral", "c2d_strength": None, "stance": "neutral"}

results = []
for i, row in df.iterrows():
    out = label_row(row["text"])
    results.append(out)
    if (len(results) % 10) == 0:
        print(f"  {len(results)}/{len(df)} done...")
    time.sleep(0.1)

df["v2_buyin"]        = [r.get("buyin",  "neutral") for r in results]
df["v2_c2d_strength"] = [r.get("c2d_strength") for r in results]
df["v2_stance"]       = [r.get("stance", "neutral") for r in results]
df.to_csv(OUT, index=False)
print(f"\nResults saved → {OUT}\n")

# ── Buyin evaluation ──────────────────────────────────────────────────────────
print("=" * 60)
print("BUYIN (committed / uncommitted / neutral)")
print("=" * 60)
y_true_b = df["human_label"].str.strip()
y_pred_b = df["v2_buyin"].str.strip()
k_b = cohen_kappa_score(y_true_b, y_pred_b)
f1_b = f1_score(y_true_b, y_pred_b, average="macro", zero_division=0)
print(f"κ = {k_b:.3f}   macro-F1 = {f1_b:.3f}\n")
print(classification_report(y_true_b, y_pred_b, zero_division=0))
labels_b = sorted(y_true_b.unique())
print("Confusion matrix (rows=human, cols=model):")
print(pd.DataFrame(
    confusion_matrix(y_true_b, y_pred_b, labels=labels_b),
    index=labels_b, columns=labels_b
).to_string())

# ── Stance evaluation ─────────────────────────────────────────────────────────
if "human_stance" in df.columns:
    df_s = df[df["human_stance"].str.strip() != ""].copy()
    if len(df_s) > 0:
        print("\n" + "=" * 60)
        print("STANCE (supportive / critical / neutral)")
        print("=" * 60)
        y_true_s = df_s["human_stance"].str.strip()
        y_pred_s = df_s["v2_stance"].str.strip()
        k_s = cohen_kappa_score(y_true_s, y_pred_s)
        f1_s = f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)
        print(f"κ = {k_s:.3f}   macro-F1 = {f1_s:.3f}\n")
        print(classification_report(y_true_s, y_pred_s, zero_division=0))
        labels_s = sorted(y_true_s.unique())
        print("Confusion matrix (rows=human, cols=model):")
        print(pd.DataFrame(
            confusion_matrix(y_true_s, y_pred_s, labels=labels_s),
            index=labels_s, columns=labels_s
        ).to_string())

# ── c2d_strength breakdown ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("c2d_strength breakdown (committed rows only)")
print("=" * 60)
committed_rows = df[df["v2_buyin"] == "committed"]
print(committed_rows["v2_c2d_strength"].value_counts().to_string())
print(f"Total committed predicted: {len(committed_rows)}")

# ── Disagreements for review ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BUYIN DISAGREEMENTS (human ≠ model) — review these")
print("=" * 60)
disagree = df[df["human_label"].str.strip() != df["v2_buyin"].str.strip()]
for _, r in disagree.iterrows():
    print(f"\n  Human={r['human_label']}  →  Model={r['v2_buyin']}"
          f"  (c2d={r['v2_c2d_strength']})")
    print(f"  {r['text'][:200]}")
