import json
import subprocess
import tempfile
import sqlite3
import pandas as pd
from pathlib import Path
import time
import math

from config import BIN_PATH, MODEL_PATH, DB_PATH, BLIMP_DATA_PATH

def get_perplexity(text, skip_layers=None):
    """Прогоняет текст через C-движок и возвращает Perplexity."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_prompt, \
            tempfile.NamedTemporaryFile(mode='r', delete=False, encoding='utf-8') as temp_output:

        temp_prompt.write(" " + text)
        temp_prompt.flush()
        prompt_file = temp_prompt.name
        output_json_file = temp_output.name

    cmd = [
        str(BIN_PATH), str(MODEL_PATH),
        "--batch-mode",
        "--eval-ppl",
        "--prompt-file", prompt_file,
        "--output-json", output_json_file
    ]

    if skip_layers:
        cmd.extend(["--skip-layers", skip_layers])

    ppl = None
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if process.returncode == 0:
            with open(output_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                probs = data.get("target_probs", [])
                if probs:
                    log_probs = [math.log(max(p, 1e-10)) for p in probs]
                    ppl = math.exp(-sum(log_probs) / len(log_probs))
    finally:
        Path(prompt_file).unlink(missing_ok=True)
        Path(output_json_file).unlink(missing_ok=True)

    return ppl

def run_blimp_eval(skip_layers=None):
    if not BIN_PATH.exists():
        raise FileNotFoundError(f"Бинарник не найден: {BIN_PATH}")

    if not BLIMP_DATA_PATH.exists():
        print(f"[!] Датасет BLiMP не найден: {BLIMP_DATA_PATH}")
        return

    data = []
    with open(BLIMP_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if not data:
        return

    print(f"\n[*] Запуск оценки синтаксиса BLiMP ({len(data)} пар предложений)...")
    results = []
    correct_count = 0

    for idx, item in enumerate(data):
        start_time = time.time()

        ppl_good = get_perplexity(item["sentence_good"], skip_layers)
        ppl_bad = get_perplexity(item["sentence_bad"], skip_layers)

        status = "success"
        is_correct = False

        if ppl_good is not None and ppl_bad is not None:
            # Модель "понимает" грамматику, если PPL правильного предложения ниже
            is_correct = ppl_good < ppl_bad
            if is_correct:
                correct_count += 1
        else:
            status = "error: failed to calculate PPL"

        run_time = time.time() - start_time

        marker = "✅" if is_correct else "❌"
        print(f"  -> {idx+1:02d} | Good PPL: {ppl_good:.2f} | Bad PPL: {ppl_bad:.2f} | {marker} ({item['category']})")

        results.append({
            "task_id": idx,
            "task_type": "blimp",
            "category": item["category"],
            "ppl_good": ppl_good,
            "ppl_bad": ppl_bad,
            "is_correct": int(is_correct),
            "run_time_sec": run_time,
            "status": status
        })

    # Сохраняем в БД
    df = pd.DataFrame(results)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("baseline_blimp", conn, if_exists="replace", index=False)
    conn.close()

    # Детализация
    print("\n--- Детализация грамматики BLiMP ---")
    category_accuracy = df.groupby('category')['is_correct'].mean() * 100
    for cat, acc in category_accuracy.items():
        print(f"Категория '{cat}': {acc:.0f}%")

    accuracy = (correct_count / len(data)) * 100 if data else 0
    print(f"\n[!] Итоговая Accuracy BLiMP: {accuracy:.2f}% ({correct_count}/{len(data)})")

