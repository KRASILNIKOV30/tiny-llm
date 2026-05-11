import json
import subprocess
import tempfile
import sqlite3
import pandas as pd
from pathlib import Path
import time
import math

from config import BIN_PATH, MODEL_PATH, DB_PATH
from data_loader import load_local_dataset

def run_wikitext_eval(skip_layers=None):
    """Прогон метрики Perplexity по локальным файлам"""
    if not BIN_PATH.exists():
        raise FileNotFoundError(f"Бинарник не найден: {BIN_PATH}")

    chunks = load_local_dataset()
    if not chunks:
        print("[!] Нет данных для тестирования. Выход.")
        return

    results = []
    print("\n[*] Запуск оценки Perplexity (Baseline)...")

    for idx, item in enumerate(chunks):
        filename = item["filename"]
        chunk_text = item["text"]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_prompt, \
                tempfile.NamedTemporaryFile(mode='r', delete=False, encoding='utf-8') as temp_output:

            temp_prompt.write(chunk_text)
            temp_prompt.flush()
            prompt_file = temp_prompt.name
            output_json_file = temp_output.name

        print(f"  -> Прогон файла {filename} ({idx+1}/{len(chunks)})...", end=" ", flush=True)
        start_time = time.time()

        # Инициализируем переменные ДО блока try
        perplexity = None
        status = "unknown"

        cmd_ppl = [
            str(BIN_PATH), str(MODEL_PATH),
            "--batch-mode",
            "--eval-ppl",
            "--prompt-file", prompt_file,
            "--output-json", output_json_file
        ]

        if skip_layers:
            cmd_ppl.extend(["--skip-layers", skip_layers])

        try:
            process = subprocess.run(cmd_ppl, capture_output=True, text=True, check=False)

            if process.returncode == 0:
                with open(output_json_file, 'r', encoding='utf-8') as f:
                    ppl_data = json.load(f)
                    probs = ppl_data.get("target_probs", [])

                    if probs:
                        log_probs = [math.log(max(p, 1e-10)) for p in probs]
                        perplexity = math.exp(-sum(log_probs) / len(log_probs))
                        status = "success"
                    else:
                        status = "no_probs (пустой JSON или нет токенов)"
            else:
                status = f"error: {process.stderr.strip()[:50]}"

        except Exception as e:
            status = f"crash: {str(e)}"

        finally:
            Path(prompt_file).unlink(missing_ok=True)
            Path(output_json_file).unlink(missing_ok=True)

        run_time = time.time() - start_time

        # Безопасное форматирование строки
        ppl_str = f"{perplexity:.2f}" if perplexity is not None else "N/A"
        print(f"ОК (Время: {run_time:.2f}с, PPL: {ppl_str} | Статус: {status})")

        results.append({
            "filename": filename,
            "task_type": "wikitext_ppl",
            "perplexity": perplexity,
            "run_time_sec": run_time,
            "status": status
        })

    df = pd.DataFrame(results)

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("baseline_wikitext", conn, if_exists="replace", index=False)
    conn.close()

    valid_ppl = df[df['status'] == 'success']['perplexity']
    if not valid_ppl.empty:
        print(f"\n[!] Итоговая средняя Perplexity корпуса: {valid_ppl.mean():.2f}")
    else:
        print("\n[!] Не удалось получить Perplexity ни для одного файла. Проверьте C-движок.")