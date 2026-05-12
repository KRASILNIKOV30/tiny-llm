import json
import subprocess
import tempfile
import sqlite3
import pandas as pd
from pathlib import Path
import time

from config import BIN_PATH, MODEL_PATH, DB_PATH

# Путь для файла с полными ответами
RESPONSES_LOG_PATH = Path("./script/chatml_responses.jsonl")

def run_chatml_retention_eval(skip_layers=None, head_mask=None, mlp_mask=None, rope_mask=None, num_tests=10):
    if not BIN_PATH.exists():
        raise FileNotFoundError(f"Бинарник не найден: {BIN_PATH}")

    # Тестовые промпты разной сложности
    test_prompts = [
        "Write a short poem about C programming.",
        "Explain what RMSNorm does in 3 sentences.",
        "Tell me a joke.",
        "List 5 capital cities in Europe.",
        "How do I use Makefile for a C project?",
    ]

    results = []
    print(f"\n[*] Запуск ChatML Formatting Retention ({len(test_prompts[:num_tests])} тестов)...")

    for idx, user_query in enumerate(test_prompts[:num_tests]):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_prompt, \
                tempfile.NamedTemporaryFile(mode='r', delete=False, encoding='utf-8') as temp_output:

            temp_prompt.write(user_query)
            temp_prompt.flush()
            prompt_file = temp_prompt.name
            output_json_file = temp_output.name

        start_time = time.time()

        # Инициализируем переменные до try-блока
        response_text = ""
        score = 0.0
        fail_reason = None
        status = "unknown"

        # Запуск
        cmd = [
            str(BIN_PATH), str(MODEL_PATH),
            "--batch-mode",
            "--prompt-file", prompt_file,
            "--output-json", output_json_file,
            "--max-tokens", "300"
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

                # АНАЛИЗ ФОРМАТА
                has_illegal_start = "<|im_start|>" in response_text
                has_proper_end = response_text.strip().endswith("<|im_end|>")

                score = 1.0

                if has_illegal_start:
                    score -= 0.5
                    fail_reason = "Illegal <|im_start|> found"
                if not has_proper_end:
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

        # 1. Сборка результатов для SQLite
        results.append({
            "task_type": "chatml_retention",
            "prompt": user_query[:30],
            "retention_score": score,
            "fail_reason": fail_reason,
            "status": status,
            "layer_mask": skip_layers if skip_layers else "None",
            "head_mask": head_mask if head_mask else "None",
            "mlp_mask": mlp_mask if mlp_mask else "None",
            "rope_mask": rope_mask if rope_mask else "None",
        })

        # 2. Сохранение полных ответов и параметров в JSONL для ИИ
        log_entry = {
            "prompt": user_query,
            "response": response_text,
            "ablation": {
                "layer_mask": skip_layers if skip_layers else "None",
                "head_mask": head_mask if head_mask else "None",
                "mlp_mask": mlp_mask if mlp_mask else "None",
                "rope_mask": rope_mask if rope_mask else "None",
            },
            "metrics": {
                "score": score,
                "status": status,
                "fail_reason": fail_reason
            }
        }

        # Режим 'a' (append) дописывает в файл, не стирая старое
        with open(RESPONSES_LOG_PATH, 'a', encoding='utf-8') as log_file:
            log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # Сохранение в БД
    df = pd.DataFrame(results)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("baseline_chatml_retention", conn, if_exists="append", index=False)
    conn.close()

    print(f"\n[!] Итоговый ChatML Retention Score: {df['retention_score'].mean()*100:.1f}%")
    print(f"[+] Логи ответов сохранены в {RESPONSES_LOG_PATH}")

if __name__ == "__main__":
    run_chatml_retention_eval()