import os, time, json
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
        "Reply with only A or B.\n\n"
        f"Story context:\n{context}\n\n"
        f"A: {e1}\n"
        f"B: {e2}"
    )

# Resume support
done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["story_id"])
print(f"Already done: {len(done)}")

correct = total = errors = 0

# Tally already-done results
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
            pred_text = resp.choices[0].message.content.strip().upper()
            pred_int  = 1 if "A" in pred_text else 2
            is_correct = int(pred_int == answer)
            correct   += is_correct
            total     += 1
            fout.write(json.dumps({
                "story_id": idx, "pred": pred_int,
                "answer": answer, "correct": is_correct
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
print(f"\n=== Zero-shot GPT-4o | Val B ===")
print(f"Accuracy : {correct}/{total} = {acc:.4f}")
print(f"p-value  : {res.pvalue:.4f}  ({'significant' if res.pvalue < 0.05 else 'not significant'})")
print(f"Errors   : {errors}")
