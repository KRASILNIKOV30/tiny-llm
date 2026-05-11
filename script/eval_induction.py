import json
import subprocess
import tempfile
import sqlite3
import pandas as pd
from pathlib import Path
import random

from config import BIN_PATH, MODEL_PATH, DB_PATH, SAFE_WORDS

def run_induction_eval(skip_layers=None, head_mask=None, mlp_mask=None, rope_mask=None, num_samples=10, seq_len=15):
    if not BIN_PATH.exists():
        raise FileNotFoundError(f"Бинарник не найден: {BIN_PATH}")

    results = []
    print(f"\n[*] Запуск оценки Induction Heads ({num_samples} прогонов)...")

    for i in range(num_samples):
        seq = random.choices(SAFE_WORDS, k=seq_len)
        text = "".join(seq) + "".join(seq)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_prompt, \
                tempfile.NamedTemporaryFile(mode='r', delete=False, encoding='utf-8') as temp_output:

            temp_prompt.write(text)
            temp_prompt.flush()
            prompt_file = temp_prompt.name
            output_json_file = temp_output.name

        # === ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ ДО TRY ===
        prob_first = 0.0
        prob_second = 0.0
        induction_score = 0.0
        status = "unknown"

        cmd_ppl = [
            str(BIN_PATH), str(MODEL_PATH),
            "--batch-mode",
            "--eval-ppl",
            "--prompt-file", prompt_file,
            "--output-json", output_json_file
        ]

        if skip_layers: cmd_ppl.extend(["--skip-layers", skip_layers])
        if head_mask: cmd_ppl.extend(["--mask-head", head_mask])
        if mlp_mask: cmd_ppl.extend(["--mask-mlp", mlp_mask])
        if rope_mask: cmd_ppl.extend(["--mask-rope", rope_mask])

        try:
            process = subprocess.run(cmd_ppl, capture_output=True, text=True, check=False)

            if process.stderr and "[Ablation]" in process.stderr:
                print(f" {process.stderr.strip()}", end="")

            if process.returncode == 0:
                with open(output_json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    probs = data.get("target_probs", [])

                    if len(probs) > 10:
                        mid = len(probs) // 2
                        first_half = probs[:mid]
                        second_half = probs[mid:]

                        prob_first = sum(first_half) / len(first_half)
                        prob_second = sum(second_half) / len(second_half)
                        induction_score = prob_second - prob_first
                        status = "success"
                    else:
                        status = "too_few_tokens"
            else:
                status = f"error: {process.stderr.strip()[:50]}"

        except Exception as e:
            status = f"crash: {str(e)}"

        finally:
            Path(prompt_file).unlink(missing_ok=True)
            Path(output_json_file).unlink(missing_ok=True)

        print(f"  -> Прогон {i+1:02d} | Вероятность 1-й части: {prob_first*100:05.2f}% | 2-й части: {prob_second*100:05.2f}% | Score: {induction_score:+.4f}")

        results.append({
            "task_id": i,
            "task_type": "induction_heads",
            "prob_first_half": prob_first,
            "prob_second_half": prob_second,
            "induction_score": induction_score,
            "status": status,
            "layer_mask": skip_layers if skip_layers else "None",
            "head_mask": head_mask if head_mask else "None",
            "mlp_mask": mlp_mask if mlp_mask else "None",
            "rope_mask": rope_mask if rope_mask else "None",
        })

    df = pd.DataFrame(results)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("baseline_induction", conn, if_exists="replace", index=False)
    conn.close()

    valid_scores = df[df['status'] == 'success']['induction_score']
    if not valid_scores.empty:
        print(f"\n[!] Итоговый средний Induction Score: {valid_scores.mean():.4f}")