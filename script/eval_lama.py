import json
import subprocess
import tempfile
import sqlite3
import pandas as pd
from pathlib import Path
import time

from config import BIN_PATH, MODEL_PATH, DB_PATH, LAMA_DATA_PATH

def load_lama_data():
    if not LAMA_DATA_PATH.exists():
        print(f"[!] Датасет LAMA не найден: {LAMA_DATA_PATH}")
        return []

    data = []
    with open(LAMA_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def run_lama_eval(skip_layers=None):
    if not BIN_PATH.exists():
        raise FileNotFoundError(f"Бинарник не найден: {BIN_PATH}")

    questions = load_lama_data()
    if not questions:
        return

    results = []
    print(f"\n[*] Запуск оценки фактологии LAMA ({len(questions)} промптов)...")

    correct_count = 0

    for idx, item in enumerate(questions):
        # Формируем промпт. В LAMA мы НЕ используем ChatML-теги!
        # Мы просто даем сырой текст, чтобы модель его продолжила.
        prompt_text = item["prompt"]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_prompt, \
                tempfile.NamedTemporaryFile(mode='r', delete=False, encoding='utf-8') as temp_output:

            temp_prompt.write(prompt_text)
            temp_prompt.flush()
            prompt_file = temp_prompt.name
            output_json_file = temp_output.name

        start_time = time.time()
        status = "unknown"
        predicted_token = ""
        is_correct = False

        # Вызываем бинарник с флагом --eval-lama
        cmd = [
            str(BIN_PATH), str(MODEL_PATH),
            "--batch-mode",
            "--eval-lama",
            "--prompt-file", prompt_file,
            "--output-json", output_json_file
        ]

        if skip_layers:
            cmd.extend(["--skip-layers", skip_layers])

        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if process.returncode == 0:
                with open(output_json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # C-движок записывает предсказанный токен в "response"
                    predicted_token = data.get("response", "")

                    # Очищаем от пробелов, приводим к нижнему регистру
                    pred_clean = predicted_token.strip().lower()
                    target_clean = item['target'].strip().lower()

                    # Токен может быть чуть шире (например, "paris." или " paris"), проверяем вхождение
                    if target_clean in pred_clean or pred_clean == target_clean:
                        is_correct = True
                        correct_count += 1

                    status = "success"
            else:
                status = f"error: {process.stderr.strip()[:50]}"

        except Exception as e:
            status = f"crash: {str(e)}"

        finally:
            Path(prompt_file).unlink(missing_ok=True)
            Path(output_json_file).unlink(missing_ok=True)

        run_time = time.time() - start_time

        marker = "✅" if is_correct else "❌"
        print(f"  -> {idx+1:02d} | Промпт: '{prompt_text}' | Ожидалось: '{item['target']}' | Токен: '{predicted_token}' {marker}")

        results.append({
            "task_id": idx,
            "task_type": "lama",
            "category": item.get("category", "general"),
            "expected": item['target'],
            "predicted": predicted_token,
            "is_correct": int(is_correct),
            "run_time_sec": run_time,
            "status": status
        })

    # Сохраняем результаты
    df = pd.DataFrame(results)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("baseline_lama", conn, if_exists="replace", index=False)
    conn.close()

    print("\n--- Детализация фактологии LAMA ---")
    category_accuracy = df.groupby('category')['is_correct'].mean() * 100
    for cat, acc in category_accuracy.items():
        print(f"Категория '{cat}': {acc:.2f}%")

    accuracy = (correct_count / len(questions)) * 100 if questions else 0
    print(f"\n[!] Итоговая Accuracy LAMA: {accuracy:.2f}% ({correct_count}/{len(questions)})")
