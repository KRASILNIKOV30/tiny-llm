import json
import subprocess
import tempfile
import sqlite3
import pandas as pd
from pathlib import Path
import time

from config import BIN_PATH, MODEL_PATH, DB_PATH, MCQA_DATA_PATH

def load_mcqa_data():
    if not MCQA_DATA_PATH.exists():
        print(f"[!] Файл с вопросами не найден: {MCQA_DATA_PATH}")
        return []

    data = []
    with open(MCQA_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def run_mcqa_eval(skip_layers=None, head_mask=None, mlp_mask=None, rope_mask=None):
    if not BIN_PATH.exists():
        raise FileNotFoundError(f"Бинарник не найден: {BIN_PATH}")

    questions = load_mcqa_data()
    if not questions:
        return

    results = []
    print(f"\n[*] Запуск оценки MCQA через логиты ({len(questions)} вопросов)...")

    correct_count = 0

    for idx, item in enumerate(questions):
        # Вручную оборачиваем в ChatML формат
        prompt_text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{item['question']}\n"
            f"A) {item['A']}\n"
            f"B) {item['B']}\n"
            f"C) {item['C']}\n"
            f"D) {item['D']}\n"
            f"Answer with only the letter of the correct option.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_prompt, \
                tempfile.NamedTemporaryFile(mode='r', delete=False, encoding='utf-8') as temp_output:

            temp_prompt.write(prompt_text)
            temp_prompt.flush()
            prompt_file = temp_prompt.name
            output_json_file = temp_output.name

        start_time = time.time()
        status = "unknown"
        predicted_answer = None
        is_correct = False

        # Запуск C-движка с новым флагом --eval-mcqa
        cmd = [
            str(BIN_PATH), str(MODEL_PATH),
            "--batch-mode",
            "--eval-mcqa",
            "--prompt-file", prompt_file,
            "--output-json", output_json_file
        ]

        if skip_layers: cmd.extend(["--skip-layers", skip_layers])
        if head_mask: cmd.extend(["--mask-head", head_mask])
        if mlp_mask: cmd.extend(["--mask-mlp", mlp_mask])
        if rope_mask: cmd.extend(["--mask-rope", rope_mask])

        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if process.returncode == 0:
                with open(output_json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # Забираем уже готового победителя, вычисленного на C-бэкенде
                    predicted_answer = data.get("prediction", "N/A")
                    is_correct = (predicted_answer == item['answer'])

                    if is_correct:
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
        print(f"  -> Прогон {idx+1:03d} | Ожидалось: {item['answer']} | Логит-победитель: {predicted_answer} {marker} | Время: {run_time:.2f}с")

        results.append({
            "task_id": idx,
            "task_type": "mcqa",
            "category": item.get('category', 'general'),
            "expected": item['answer'],
            "predicted": predicted_answer,
            "is_correct": int(is_correct),
            "run_time_sec": run_time,
            "status": status,
            "layer_mask": skip_layers if skip_layers else "None",
            "head_mask": head_mask if head_mask else "None",
            "mlp_mask": mlp_mask if mlp_mask else "None",
            "rope_mask": rope_mask if rope_mask else "None",
        })

    # Сохранение результатов в БД
    df = pd.DataFrame(results)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("baseline_mcqa_logits", conn, if_exists="replace", index=False)
    conn.close()

    # Детальная статистика
    print("\n--- Детализация деградации ---")
    category_accuracy = df.groupby('category')['is_correct'].mean() * 100
    for cat, acc in category_accuracy.items():
        print(f"Категория '{cat}': {acc:.2f}%")

    accuracy = (correct_count / len(questions)) * 100 if questions else 0
    print(f"\n[!] Итоговая Accuracy MCQA: {accuracy:.2f}% ({correct_count}/{len(questions)})")

if __name__ == "__main__":
    run_mcqa_eval()