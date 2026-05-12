import os, re, time, json
import pandas as pd
from openai import OpenAI
from scipy.stats import binomtest
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DATA = "data/raw/cloze_test_val__winter2018-cloze_test_ALL_val.csv"
OUT  = "data/processed/zero_shot_gpt4o_results.jsonl"

cloze_val = pd.read_csv(DATA)
print(f"Loaded {len(cloze_val)} stories")

def zero_shot_prompt(context, e1, e2):
    return (
        "You are given a short story with four sentences of context and two possible endings.\n"
        "Choose the ending that makes more sense as a continuation of the story.\n"
        "Reply with only the letter A or B.\n\n"
        f"Story context:\n{context}\n\n"
        f"A: {e1}\n"
        f"B: {e2}"
    )

def parse_pred(raw: str) -> int:
    """Return 1 (A) or 2 (B). Robust to 'A.', 'Answer: B', etc."""
    text = raw.strip().upper()
    # exact single char
    if text in ("A", "B"):
        return 1 if text == "A" else 2
    # first standalone A or B
    m = re.search(r'\b([AB])\b', text)
    if m:
        return 1 if m.group(1) == "A" else 2
    # fallback: first character
    if text and text[0] in ("A", "B"):
        return 1 if text[0] == "A" else 2
    return -1  # unparseable

# Resume support
done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["story_id"])
print(f"Already done: {len(done)}")

correct = total = errors = unparsed = 0
if done:
    with open(OUT) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                correct += int(r["correct"])
                total   += 1

with open(OUT, "a") as fout:
    for idx, row in cloze_val.iterrows():
        if idx in done:
            continue
        context = " ".join([str(row[f"InputSentence{i}"]) for i in range(1, 5)])
        e1      = str(row["RandomFifthSentenceQuiz1"])
        e2      = str(row["RandomFifthSentenceQuiz2"])
        answer  = int(row["AnswerRightEnding"])
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": zero_shot_prompt(context, e1, e2)}],
                max_tokens=5,
                temperature=0,
            )
            raw_text = resp.choices[0].message.content
            pred_int = parse_pred(raw_text)
            if pred_int == -1:
                unparsed += 1
                print(f"Unparseable at {idx}: {raw_text!r}")
                pred_int = 2  # default fallback
            is_correct = int(pred_int == answer)
            correct   += is_correct
            total     += 1
            fout.write(json.dumps({
                "story_id": idx, "pred": pred_int, "answer": answer,
                "correct": is_correct, "raw": raw_text
            }) + "\n")
            fout.flush()
            if total % 100 == 0:
                print(f"  {total}/1571 — running acc: {correct/total:.4f}")
        except Exception as e:
            errors += 1
            print(f"Error at {idx}: {e}")
        time.sleep(0.3)

acc = correct / total if total > 0 else 0
res = binomtest(correct, total, 0.5, alternative="greater")

# Position bias check
a_when_correct = b_when_correct = a_total = b_total = 0
with open(OUT) as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        if r["answer"] == 1:
            a_total += 1
            a_when_correct += r["correct"]
        else:
            b_total += 1
            b_when_correct += r["correct"]

print(f"\n=== Zero-shot GPT-4o | Val B ===")
print(f"Accuracy  : {correct}/{total} = {acc:.4f}")
print(f"p-value   : {res.pvalue:.4e}  ({'significant' if res.pvalue < 0.05 else 'not significant'})")
print(f"Errors    : {errors}  |  Unparsed: {unparsed}")
print(f"\nPosition bias check:")
print(f"  Acc when answer=A : {a_when_correct}/{a_total} = {a_when_correct/a_total:.4f}")
print(f"  Acc when answer=B : {b_when_correct}/{b_total} = {b_when_correct/b_total:.4f}")
