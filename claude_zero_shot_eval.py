"""Zero-shot Claude evaluation on Val B and Val C."""
import os, re, time, json
import pandas as pd
import anthropic
from scipy.stats import binomtest
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL  = "claude-opus-4-7"

DATA_VALB = "data/raw/cloze_test_val__winter2018-cloze_test_ALL_val.csv"
DATA_VALC = "data/processed/cloze_negatives.jsonl"
OUT_VALB  = "data/processed/zero_shot_claude_valb.jsonl"
OUT_VALC  = "data/processed/zero_shot_claude_valc.jsonl"

def zero_shot_prompt(context, e1, e2):
    return (
        "You are given a short story with four sentences of context and two possible endings.\n"
        "Choose the ending that makes more sense as a continuation of the story.\n"
        "Reply with only the letter A or B.\n\n"
        f"Story context:\n{context}\n\n"
        f"A: {e1}\n"
        f"B: {e2}"
    )

def parse_pred(raw):
    text = raw.strip().upper()
    if text in ("A", "B"):
        return 1 if text == "A" else 2
    m = re.search(r'\b([AB])\b', text)
    if m:
        return 1 if m.group(1) == "A" else 2
    if text and text[0] in ("A", "B"):
        return 1 if text[0] == "A" else 2
    return -1

def call_claude(prompt):
    msg = client.messages.create(
        model=MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

def position_bias(out_path):
    a_correct = a_total = b_correct = b_total = 0
    with open(out_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["answer"] == 1:
                a_total   += 1
                a_correct += r["correct"]
            else:
                b_total   += 1
                b_correct += r["correct"]
    print(f"  Acc when answer=A: {a_correct}/{a_total} = {a_correct/a_total:.4f}")
    print(f"  Acc when answer=B: {b_correct}/{b_total} = {b_correct/b_total:.4f}")

# ── Val B ──────────────────────────────────────────────────────────────────
print("=== Val B ===")
cloze_val = pd.read_csv(DATA_VALB)

done_b = set()
if os.path.exists(OUT_VALB):
    with open(OUT_VALB) as f:
        for line in f:
            if line.strip():
                done_b.add(json.loads(line)["story_id"])
print(f"Already done: {len(done_b)}")

correct_b = total_b = errors_b = unparsed_b = 0
if done_b:
    with open(OUT_VALB) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                correct_b += r["correct"]
                total_b   += 1

with open(OUT_VALB, "a") as fout:
    for idx, row in cloze_val.iterrows():
        if idx in done_b:
            continue
        context = " ".join([str(row[f"InputSentence{i}"]) for i in range(1, 5)])
        e1      = str(row["RandomFifthSentenceQuiz1"])
        e2      = str(row["RandomFifthSentenceQuiz2"])
        answer  = int(row["AnswerRightEnding"])
        try:
            raw      = call_claude(zero_shot_prompt(context, e1, e2))
            pred_int = parse_pred(raw)
            if pred_int == -1:
                unparsed_b += 1
                print(f"  Unparseable at {idx}: {raw!r}")
                pred_int = 2
            is_correct  = int(pred_int == answer)
            correct_b  += is_correct
            total_b    += 1
            fout.write(json.dumps({
                "story_id": idx, "pred": pred_int, "answer": answer,
                "correct": is_correct, "raw": raw
            }) + "\n")
            fout.flush()
            if total_b % 100 == 0:
                print(f"  {total_b}/1571 — acc: {correct_b/total_b:.4f}")
        except Exception as e:
            errors_b += 1
            print(f"  Error at {idx}: {e}")
        time.sleep(0.3)

acc_b = correct_b / total_b if total_b > 0 else 0
res_b = binomtest(correct_b, total_b, 0.5, alternative="greater")
print(f"\nVal B: {correct_b}/{total_b} = {acc_b:.4f}  p={res_b.pvalue:.2e}"
      f"  ({'sig' if res_b.pvalue < 0.05 else 'not sig'})")
print(f"Errors: {errors_b}  Unparsed: {unparsed_b}")
print("Position bias:")
position_bias(OUT_VALB)

# ── Val C ──────────────────────────────────────────────────────────────────
if not os.path.exists(DATA_VALC):
    print(f"\nVal C data not found at {DATA_VALC} — run regen_cloze_negatives.py first")
else:
    print("\n=== Val C ===")
    cloze_neg = []
    with open(DATA_VALC) as f:
        for line in f:
            if line.strip():
                cloze_neg.append(json.loads(line))
    print(f"Loaded {len(cloze_neg)} stories for Val C")

    done_c = set()
    if os.path.exists(OUT_VALC):
        with open(OUT_VALC) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_c.add((r["story_id"], r["type"]))

    type_labels = ["temporal", "causal", "state", "sentiment"]
    type_correct = {t: 0 for t in type_labels}
    type_total   = {t: 0 for t in type_labels}

    # Tally already done
    if done_c:
        with open(OUT_VALC) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    type_correct[r["type"]] += r["correct"]
                    type_total[r["type"]]   += 1

    total_c = errors_c = 0
    with open(OUT_VALC, "a") as fout:
        for row in cloze_neg:
            sid     = row["story_id"]
            context = row["context"]
            gold    = row["gold_ending"]
            for neg_type in type_labels:
                if (sid, neg_type) in done_c:
                    continue
                neg = row.get(f"{neg_type}_negative", "")
                if not neg:
                    continue
                # Randomise position to control for bias
                import random
                if random.random() < 0.5:
                    e1, e2, answer = gold, neg, 1
                else:
                    e1, e2, answer = neg, gold, 2
                try:
                    raw      = call_claude(zero_shot_prompt(context, e1, e2))
                    pred_int = parse_pred(raw)
                    if pred_int == -1:
                        pred_int = 2
                    is_correct = int(pred_int == answer)
                    type_correct[neg_type] += is_correct
                    type_total[neg_type]   += 1
                    total_c += 1
                    fout.write(json.dumps({
                        "story_id": sid, "type": neg_type,
                        "pred": pred_int, "answer": answer,
                        "correct": is_correct, "raw": raw
                    }) + "\n")
                    fout.flush()
                    if total_c % 200 == 0:
                        print(f"  {total_c} pairs done")
                except Exception as e:
                    errors_c += 1
                    print(f"  Error ({neg_type}, {sid}): {e}")
                time.sleep(0.25)

    print(f"\nVal C results (n≈{type_total.get('temporal',0)} per type):")
    overall_c = 0
    overall_n = 0
    for t in type_labels:
        n   = type_total[t]
        acc = type_correct[t] / n if n > 0 else 0
        res = binomtest(type_correct[t], n, 0.5, alternative="greater") if n > 0 else None
        p   = res.pvalue if res else float("nan")
        overall_c += type_correct[t]
        overall_n += n
        print(f"  {t:<12} {acc:.4f}  p={p:.2e}")
    if overall_n:
        print(f"  {'overall':<12} {overall_c/overall_n:.4f}")
    print(f"Errors: {errors_c}")
