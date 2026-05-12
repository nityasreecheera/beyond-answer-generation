"""Regenerate cloze_negatives.jsonl for Val C evaluation."""
import os, re, time, json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DATA = "data/raw/cloze_test_val__winter2018-cloze_test_ALL_val.csv"
OUT  = "data/processed/cloze_negatives.jsonl"
ERR  = "data/processed/cloze_negatives_errors.jsonl"

cloze_val = pd.read_csv(DATA).reset_index(drop=True)
cloze_val["context"] = cloze_val.apply(
    lambda r: " ".join([str(r[f"InputSentence{i}"]) for i in range(1, 5)]), axis=1)
cloze_val["gold_ending"] = cloze_val.apply(
    lambda r: str(r["RandomFifthSentenceQuiz1"]) if r["AnswerRightEnding"] == 1
              else str(r["RandomFifthSentenceQuiz2"]), axis=1)
cloze_val["story_id"] = cloze_val.index
print(f"Loaded {len(cloze_val)} Cloze Val stories")

# Resume support
done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["story_id"])
print(f"Already done: {len(done)}")

def build_prompt(neg_type, context, gold_ending):
    rules = {
        "temporal": (
            "is clearly WRONG only because it creates a TEMPORAL inconsistency\n"
            "- must violate the expected order, timing, or duration of events\n"
            "- should make something happen too early, too late, or in the wrong sequence\n"
            "- must not mainly be a sentiment flip or causal contradiction"
        ),
        "causal": (
            "is clearly WRONG only because it creates a CAUSAL inconsistency\n"
            "- must directly contradict what should logically happen because of the earlier events\n"
            "- the outcome must clearly contradict the expected result of earlier actions\n"
            "- must not mainly be a timing/order mistake or emotion reversal"
        ),
        "state": (
            "is clearly WRONG only because it creates a STATE inconsistency\n"
            "- must contradict the established world state, condition, possession, location, or fact\n"
            "- must not mainly be a timing/order mistake or causal contradiction"
        ),
        "sentiment": (
            "is clearly WRONG only because it creates a SENTIMENT contradiction\n"
            "- must reverse or conflict with the emotional direction of the story\n"
            "- must not mainly be a timing/order mistake or causal contradiction"
        ),
    }
    return (
        "You are creating a single-sentence incorrect ending for a narrative reasoning dataset.\n\n"
        "Write ONE ending that:\n"
        "- uses the same main character(s), setting, and topic\n"
        "- is fluent, natural, and realistic\n"
        f"- {rules[neg_type]}\n"
        "- must stay close in topic to the correct ending\n"
        "- must not introduce new named characters or locations\n"
        "- must be exactly one sentence\n\n"
        f"Context:\n{context}\n\n"
        f"Correct ending:\n{gold_ending}\n\n"
        "Return only the incorrect ending sentence."
    )

def generate_one(prompt):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.8,
    )
    return resp.choices[0].message.content.strip()

def is_valid(text, gold):
    words = text.split()
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return (4 <= len(words) <= 25
            and len(sentences) == 1
            and text.lower() != gold.lower())

SLEEP = 0.4
total = errors = 0

with open(OUT, "a") as fout, open(ERR, "a") as ferr:
    for _, row in cloze_val.iterrows():
        sid = int(row["story_id"])
        if sid in done:
            continue
        context = row["context"]
        gold    = row["gold_ending"]
        result  = {"story_id": sid, "context": context, "gold_ending": gold}
        ok = True
        for neg_type in ["temporal", "causal", "state", "sentiment"]:
            try:
                text = generate_one(build_prompt(neg_type, context, gold))
                if not is_valid(text, gold):
                    text = generate_one(build_prompt(neg_type, context, gold))  # one retry
                result[f"{neg_type}_negative"] = text
                time.sleep(SLEEP)
            except Exception as e:
                ferr.write(json.dumps({"story_id": sid, "error": str(e)}) + "\n")
                ferr.flush()
                errors += 1
                ok = False
                break
        if ok:
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            total += 1
            if total % 100 == 0:
                print(f"  {total}/1571 generated")

print(f"\nDone: {total} stories, {errors} errors")
print(f"Saved to {OUT}")
