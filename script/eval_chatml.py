import json
import subprocess
import tempfile
import sqlite3
import pandas as pd
from pathlib import Path
import time

from config import BIN_PATH, MODEL_PATH, DB_PATH

def run_chatml_retention_eval(skip_layers=None, num_tests=10):
    if not BIN_PATH.exists():
        raise FileNotFoundError(f"Бинарник не найден: {BIN_PATH}")

    # Тестовые промпты разной сложности
    test_prompts = [
        "Write a short poem about C programming.",
        "Explain what RMSNorm does in 3 sentences.",
        "Tell me a joke.",
        "List 5 capital cities in Europe.",
        "How do I use Makefile for a C project?",
        "What is the difference between Q and K heads in GQA?",
        "Write a hello world in Python.",
        "Who is Isaac Newton?",
        "Summarize the benefits of GGUF format.",
        "Give me a recipe for a simple cake."
    ]

    results = []
    print(f"\n[*] Запуск ChatML Formatting Retention ({len(test_prompts[:num_tests])} тестов)...")

    for idx, user_query in enumerate(test_prompts[:num_tests]):
        # Формируем запрос (бинарник внутри сам обернет это в ChatML через chat_format_delta)
        # Но мы проверяем, как модель завершает генерацию.

        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_prompt, \
                tempfile.NamedTemporaryFile(mode='r', delete=False, encoding='utf-8') as temp_output:

            temp_prompt.write(user_query)
            temp_prompt.flush()
            prompt_file = temp_prompt.name
            output_json_file = temp_output.name

        start_time = time.time()

        # Запуск
        cmd = [
            str(BIN_PATH), str(MODEL_PATH),
            "--batch-mode",
            "--prompt-file", prompt_file,
            "--output-json", output_json_file,
            "--max-tokens", "512"
        ]

        if skip_layers:
            cmd.extend(["--skip-layers", skip_layers])

        try:
            # Нам нужно поймать именно то, что модель выдала в ответ
            process = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if process.returncode == 0:
                with open(output_json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    response = data.get("response", "")

                # АНАЛИЗ ФОРМАТА
                # 1. Проверка на наличие лишних открывающих тегов (модель не должна сама их писать)
                has_illegal_start = "<|im_start|>" in response

                # 2. Проверка на корректное завершение.
                # Т.к. твой движок в engine_generate проверяет EOS,
                # мы смотрим, добавил ли колбэк тег из токенизатора.
                has_proper_end = response.strip().endswith("<|im_end|>")

                # В Qwen2.5 EOS токен обычно декодируется как <|im_end|> или вызывает остановку.
                # Если в response_text есть <|im_end|>, значит retention = 1.0
                score = 1.0
                fail_reason = None

                if has_illegal_start:
                    score -= 0.5
                    fail_reason = "Illegal <|im_start|> found"
                if not has_proper_end:
                    # Примечание: если движок обрезает EOS до записи в буфер,
                    # проверь логику в src/inference.c:token_cb
                    score -= 0.5
                    if fail_reason: fail_reason += " & Missing <|im_end|>"
                    else: fail_reason = "Missing <|im_end|>"

                status = "success"
            else:
                score = 0.0
                status = f"error: {process.stderr.strip()[:50]}"
                fail_reason = "Engine error"

        except Exception as e:
            score = 0.0
            status = f"crash: {str(e)}"
            fail_reason = "Evaluation crash"
        finally:
            Path(prompt_file).unlink(missing_ok=True)
            Path(output_json_file).unlink(missing_ok=True)

        print(f"  -> Тест {idx+1:02d} | Score: {score:.2f} | Status: {status} | Reason: {fail_reason}")
        print('\n' + response + '\n')

        results.append({
            "task_type": "chatml_retention",
            "prompt": user_query[:30],
            "retention_score": score,
            "fail_reason": fail_reason,
            "status": status
        })

    # Сохранение в БД
    df = pd.DataFrame(results)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("baseline_chatml_retention", conn, if_exists="replace", index=False)
    conn.close()

    print(f"\n[!] Итоговый ChatML Retention Score: {df['retention_score'].mean()*100:.1f}%")

if __name__ == "__main__":
    run_chatml_retention_eval()