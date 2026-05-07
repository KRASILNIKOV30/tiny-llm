import json
import subprocess
import tempfile
import sqlite3
import pandas as pd
from pathlib import Path
import time

# --- Настройки путей ---
BIN_PATH = Path("./bin/chat")
MODEL_PATH = Path("./model.gguf") # Замените на ваш путь
DATASET_PATH = Path("./dataset_baseline.jsonl")
DB_PATH = Path("./eval_results.db")

def generate_sample_dataset():
    """
    Шаг 1. Генерация датасета.
    Создает JSONL файл с набором тестовых промптов.
    """
    prompts = [
        {"id": 1, "task_type": "knowledge", "text": "What is the capital of France?"},
        {"id": 2, "task_type": "reasoning", "text": "If I have 3 apples and eat 1, how many are left?"},
        {"id": 3, "task_type": "coding", "text": "Write a Python function to reverse a string."},
        {"id": 4, "task_type": "translation", "text": "Translate 'Hello, how are you?' to Russian."},
        {"id": 5, "task_type": "formatting", "text": "List 3 colors in a markdown bulleted list."}
    ]

    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        for p in prompts:
            f.write(json.dumps(p) + '\n')

    print(f"[*] Датасет успешно сгенерирован: {DATASET_PATH} ({len(prompts)} примеров)")

def run_baseline_eval():
    """
    Шаг 2 и 3. Диспетчер задач и сборщик результатов.
    Читает JSONL, прогоняет через C-бинарник, собирает метрики.
    """
    if not BIN_PATH.exists():
        raise FileNotFoundError(f"Бинарник не найден: {BIN_PATH}. Соберите проект (make).")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}.")

    results = []

    # Читаем датасет
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    print(f"[*] Начинаем прогон базлайна для {len(dataset)} задач...")

    for item in dataset:
        task_id = item["id"]
        prompt_text = item["text"]
        task_type = item["task_type"]

        # Создаем временные файлы для промпта и вывода
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_prompt, \
                tempfile.NamedTemporaryFile(mode='r', delete=False, encoding='utf-8') as temp_output:

            # Записываем сырой текст промпта, как того ожидает C-движок
            temp_prompt.write(prompt_text)
            temp_prompt.flush()

            prompt_file = temp_prompt.name
            output_json_file = temp_output.name

        # Формируем команду для вызова subprocess
        cmd = [
            str(BIN_PATH),
            str(MODEL_PATH),
            "--batch-mode",
            "--prompt-file", prompt_file,
            "--output-json", output_json_file
        ]

        print(f"  -> Прогон задачи ID {task_id} [{task_type}]...", end=" ", flush=True)

        start_time = time.time()

        try:
            # Запускаем бинарник
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False # Не падаем сразу при ошибке, чтобы собрать stderr
            )

            run_time = time.time() - start_time

            # Проверяем успешность выполнения
            if process.returncode != 0:
                print("ОШИБКА!")
                print(f"     STDERR: {process.stderr.strip()}")
                response_text = f"ERROR: {process.stderr.strip()}"
            else:
                # Читаем сгенерированный JSON из временного файла
                with open(output_json_file, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                    response_text = output_data.get("response", "")
                print(f"ОК ({run_time:.2f} сек)")

        except Exception as e:
            print(f"КРАШ СКРИПТА: {e}")
            response_text = f"CRASH: {str(e)}"
            run_time = time.time() - start_time

        finally:
            # Удаляем временные файлы
            Path(prompt_file).unlink(missing_ok=True)
            Path(output_json_file).unlink(missing_ok=True)

        # Сохраняем результат
        results.append({
            "id": task_id,
            "task_type": task_type,
            "prompt": prompt_text,
            "response": response_text,
            "run_time_sec": run_time,
            "status": "success" if process.returncode == 0 else "error"
        })

    # Сохраняем все в Pandas DataFrame
    df = pd.DataFrame(results)

    # Шаг 3 (продолжение). Сохранение в SQLite базу данных
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("baseline_metrics", conn, if_exists="replace", index=False)
    conn.close()

    print(f"\n[*] Базлайн собран! Результаты сохранены в БД SQLite: {DB_PATH}")

    # Для наглядности выведем превью
    print("\nПревью результатов:")
    print(df[['id', 'task_type', 'run_time_sec', 'status']].head())

if __name__ == "__main__":
    generate_sample_dataset()
    run_baseline_eval()