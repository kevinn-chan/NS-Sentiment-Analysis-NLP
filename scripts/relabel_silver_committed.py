"""
Re-label the 2,914 LLM-silver 'committed' rows in stage2a_train.csv
using a tighter prompt that enforces first-person personal service commitment.

Root cause: old prompt allowed "see real value in it / endorse it as worth doing"
which bled into supportive STANCE texts being mislabelled as committed BUYIN.

Fix: committed = author expresses their OWN willingness/dedication/effort in
their personal service. Advice to others, policy endorsement, cheerleading,
and abstract commentary do NOT qualify.

Usage: python scripts/relabel_silver_committed.py [--dry-run] [--limit N]
Output: updates data/processed/new/cascade/stage2a_train.csv in-place
        saves backup to stage2a_train_backup.csv first
"""
import argparse, json, os, time
import pandas as pd
from openai import OpenAI
from pathlib import Path

DATA = Path("data/processed/new/cascade")
TRAIN_CSV = DATA / "stage2a_train.csv"

# ── Tightened system prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are re-labelling Reddit comments about Singapore's National Service (NS).
Your ONLY task: decide if the AUTHOR is personally committed to their own NS service.

━━━ COMMITTED (strict definition) ━━━
The author expresses, in FIRST PERSON, one or more of:
  • Personal willingness or pride in serving / defending Singapore
  • Genuine effort, discipline, or dedication in their own service
  • Personal growth they credit to their own active engagement in NS
  • Choosing to stay in or sign on out of genuine belief (not money)

COMMITTED requires: first-person + personal service + active dedication or willing mindset.

━━━ NOT COMMITTED → return "relabel_neutral" ━━━
These look positive but are NOT committed buyin:

  ADVICE TO OTHERS: "Enjoy NS, positive mentality goes a long way"
  → Author is not expressing their own commitment. Return: relabel_neutral

  CHEERLEADING / ENCOURAGEMENT: "Hope you have a great time in NS bro!"
  → Rooting for someone else, not first-person dedication. Return: relabel_neutral

  POLICY / INSTITUTIONAL SUPPORT: "NS is necessary for Singapore's defence"
  → This is stance (institutional opinion), not personal buyin. Return: relabel_neutral

  CITIZENSHIP DUTY ARGUMENT: "NS is an obligation that comes with citizenship"
  → Abstract argument about duty, not first-person commitment. Return: relabel_neutral

  RETROSPECTIVE GROWTH WITHOUT ACTIVE DEDICATION: "NS made me more confident"
  → Passive outcome, not active commitment signal. Return: relabel_neutral
  EXCEPTION: "I pushed myself hard and NS made me confident" → committed (active dedication)

  DESCRIBING OTHERS' COMMITMENT: "My platoon mate gave everything"
  → Third-person observation. Return: relabel_neutral

  GENERIC NS NARRATIVE without buy-in signal: "Went through BMT, passed IPPT"
  → Just describing events, no buy-in revealed. Return: relabel_neutral

━━━ UNCOMMITTED ━━━
The author expresses, in first person, apathy/disengagement:
  • "Just here to clear time", "zao liao", "doing bare minimum", "chao keng"
  • Counting down to ORD with no investment
  (Return: keep_uncommitted — but only if it was originally uncommitted, irrelevant here)

━━━ YOUR TASK ━━━
For each text originally labelled "committed", decide:
  "keep_committed"   — genuinely first-person dedication/willingness/pride in own service
  "relabel_neutral"  — cheerleading, advice, policy stance, passive outcome, or third-person

Reply with ONLY one of: keep_committed  OR  relabel_neutral
No explanation. No JSON. Just the label."""

# ── Contrastive few-shot examples ─────────────────────────────────────────────
FEW_SHOT = [
    # TRUE committed: first-person dedication
    ("I made sure I gave 100% every single exercise. No point half-assing something you have to do anyway — might as well be the best version of yourself.",
     "keep_committed"),
    # TRUE committed: pride in own service / willing defender
    ("I'm proud to serve. Singapore gave my family everything, the least I can do is give back 2 years to protect it.",
     "keep_committed"),
    # TRUE committed: signed on out of genuine belief
    ("I signed on not because of the pay but because I genuinely believe in what we're defending. Regulars get a bad rep but some of us are actually here for the right reasons.",
     "keep_committed"),
    # TRUE committed: active effort and growth
    ("I pushed myself during every route march even when my feet were bleeding. Came out stronger and I'm glad I didn't give up.",
     "keep_committed"),
    # MISLABELLED: cheerleading / advice
    ("Hope you have a great time in NS bro! You'd be surprised how many people are in the same situation.",
     "relabel_neutral"),
    # MISLABELLED: advice to others
    ("If you're enlisting, just enjoy NS. A positive mentality goes a long way. Make friends with your superiors but don't kowtow.",
     "relabel_neutral"),
    # MISLABELLED: citizenship duty argument (policy, not personal)
    ("Dude doesn't understand the term 'national service' and 'citizenship obligation'. This is a duty that comes with citizenship.",
     "relabel_neutral"),
    # MISLABELLED: passive growth without active dedication
    ("NS made me more open to talking to strangers and more confident in myself.",
     "relabel_neutral"),
    # MISLABELLED: institutional endorsement / policy stance
    ("NS is necessary for Singapore's survival. Small country, need to defend ourselves.",
     "relabel_neutral"),
    # MISLABELLED: describing platoon culture / others
    ("Real friendships are forged during NS times. I still meet up with my army mates.",
     "relabel_neutral"),
]

VALID = {"keep_committed", "relabel_neutral"}

client = OpenAI()

def classify(text: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_text, ex_label in FEW_SHOT:
        msgs.append({"role": "user",      "content": ex_text})
        msgs.append({"role": "assistant", "content": ex_label})
    msgs.append({"role": "user", "content": str(text)[:1500]})

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=msgs,
        max_tokens=10,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip().lower()
    return raw if raw in VALID else "relabel_neutral"  # safe default on bad output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print first 10 results, don't save")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (testing)")
    args = parser.parse_args()

    train = pd.read_csv(TRAIN_CSV)
    mask  = (train["label"] == "committed") & (train["source"] == "llm_silver")
    targets = train[mask].copy()

    if args.limit:
        targets = targets.head(args.limit)

    print(f"Re-labelling {len(targets):,} silver committed rows...")
    print(f"Estimated cost: ${len(targets) * (600*0.10 + 10*0.40) / 1_000_000:.2f}\n")

    if args.dry_run:
        for _, r in targets.head(10).iterrows():
            result = classify(r["text"])
            print(f"[{result}] {r['text'][:120]}")
        return

    # Backup
    backup = TRAIN_CSV.with_name("stage2a_train_backup.csv")
    if not backup.exists():
        train.to_csv(backup, index=False)
        print(f"Backup saved → {backup}")

    results = {}
    for i, (idx, row) in enumerate(targets.iterrows()):
        result = classify(row["text"])
        results[idx] = result
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(targets)} done...")
        time.sleep(0.02)  # stay well under rate limit

    # Apply: relabel_neutral → "neutral", keep_committed → stays "committed"
    relabelled = sum(1 for v in results.values() if v == "relabel_neutral")
    for idx, verdict in results.items():
        if verdict == "relabel_neutral":
            train.at[idx, "label"] = "neutral"

    train.to_csv(TRAIN_CSV, index=False)

    print(f"\nDone.")
    print(f"  Kept committed:    {len(targets) - relabelled:,}")
    print(f"  Relabelled neutral: {relabelled:,}  ({relabelled/len(targets):.1%})")
    print(f"\nNew label dist:")
    print(train["label"].value_counts())
    print(f"\nNext step: retrain Stage 2a SingBERT model")
    print(f"  python src/features/cascade_train_export.py --stage 2a")


if __name__ == "__main__":
    main()
