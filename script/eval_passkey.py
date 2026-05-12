import json
import subprocess
import tempfile
import sqlite3
import pandas as pd
from pathlib import Path
import random
import time

from config import BIN_PATH, MODEL_PATH, DB_PATH, SAFE_WORDS

def generate_filler_text(word_count):
    """Генерирует бессмысленный текст заданной длины."""
    return "".join(random.choices(SAFE_WORDS, k=word_count))

def run_passkey_eval(skip_layers=None, head_mask=None, mlp_mask=None, rope_mask=None, max_words=1000, steps=3):
    """
    Оценивает способность модели извлекать факт (passkey) с разной глубины контекста.
    """
    if not BIN_PATH.exists():
        raise FileNotFoundError(f"Бинарник не найден: {BIN_PATH}")

    results = []
    print(f"\n[*] Запуск оценки Passkey Retrieval (RoPE Benchmark)...")

    # Длины контекстов в словах (приблизительно = токенам)
    context_lengths = [int(x) for x in range(200, max_words + 1, max_words // steps)]

    for ctx_len in context_lengths:
        # Тестируем 3 разные глубины погружения "иголки"
        depths = [0.1, 0.5, 0.9] # 10% (начало), 50% (середина), 90% (конец)

        for depth in depths:
            passkey = random.randint(10000, 99999)
            needle = f"\n[IMPORTANT INFO: The secret passkey is {passkey}.]\n"

            # Собираем стог сена
            words_before = int(ctx_len * depth)
            words_after = ctx_len - words_before

            haystack = (
                    "There is important information hidden in this document. Find the passkey.\n" +
                    generate_filler_text(words_before) +
                    needle +
                    generate_filler_text(words_after) +
                    "\nWhat is the secret passkey? Answer with just the number."
            )

            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_prompt, \
                    tempfile.NamedTemporaryFile(mode='r', delete=False, encoding='utf-8') as temp_output:

                temp_prompt.write(haystack)
                temp_prompt.flush()
                prompt_file = temp_prompt.name
                output_json_file = temp_output.name

            start_time = time.time()
            status = "unknown"
            is_correct = False
            response_text = ""

            # Обычный chat-режим без eval-флагов, он сгенерирует ответ и положит в output_json_file
            cmd = [
                str(BIN_PATH), str(MODEL_PATH),
                "--batch-mode",
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
                        response_text = data.get("response", "")

                        # Если пароль есть в строке ответа — тест пройден
                        if str(passkey) in response_text:
                            is_correct = True
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
            print(f"  -> Длина: {ctx_len:4d} слов | Глубина: {depth*100:2.0f}% | Passkey: {passkey} | {marker} ({run_time:.1f}с)")

            results.append({
                "task_type": "passkey_retrieval",
                "context_length": ctx_len,
                "depth_pct": depth,
                "is_correct": int(is_correct),
                "run_time_sec": run_time,
                "status": status,
                "layer_mask": skip_layers if skip_layers else "None",
                "head_mask": head_mask if head_mask else "None",
                "mlp_mask": mlp_mask if mlp_mask else "None",
                "rope_mask": rope_mask if rope_mask else "None",
            })

    # Сохраняем в БД
    df = pd.DataFrame(results)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("baseline_passkey", conn, if_exists="append", index=False)
    conn.close()

    if not df.empty:
        acc = df['is_correct'].mean() * 100
        print(f"\n[!] Итоговая Accuracy Passkey Retrieval: {acc:.1f}%")

if __name__ == "__main__":
    run_passkey_eval(max_words=2000, steps=5)