import json
import sys
from pathlib import Path

def convert_dataset(input_path, output_path, category_name="logic_predicate"):
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"[!] Ошибка: Файл {input_path} не найден.")
        return

    processed_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Достаем список опций (защита от нехватки вариантов ответа)
                opts = data.get("options", [])
                while len(opts) < 4:
                    opts.append("N/A")  # Если вариантов меньше 4, добиваем пустышками

                # Формируем новую запись
                new_record = {
                    "category": category_name,
                    "question": data.get("centerpiece", "").strip(),
                    "A": str(opts[0]),
                    "B": str(opts[1]),
                    "C": str(opts[2]),
                    "D": str(opts[3]),
                    "answer": data.get("correct_options", [""])[0].strip()
                }

                outfile.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                processed_count += 1

            except json.JSONDecodeError:
                print("[!] Ошибка парсинга JSON в строке, пропускаем...")
            except Exception as e:
                print(f"[!] Непредвиденная ошибка: {e}")

    print(f"[*] Готово! Успешно конвертировано {processed_count} вопросов.")
    print(f"[*] Результат сохранен в: {output_path}")

if __name__ == "__main__":
    # Если скрипт запущен с аргументами командной строки:
    # python convert_dataset.py raw_data.jsonl mcqa_logic.jsonl
    if len(sys.argv) >= 3:
        in_file = sys.argv[1]
        out_file = sys.argv[2]
        cat_name = sys.argv[3] if len(sys.argv) > 3 else "logic_predicate"
        convert_dataset(in_file, out_file, cat_name)
    else:
        # Поведение по умолчанию для быстрого теста
        print("[*] Запуск с путями по умолчанию...")
        convert_dataset("raw_input.jsonl", "converted_output.jsonl", "logic_predicate")