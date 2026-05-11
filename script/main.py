import sqlite3
from eval_wikitext import run_wikitext_eval
from eval_induction import run_induction_eval
from eval_mcqa import run_mcqa_eval
from eval_blimp import run_blimp_eval
from eval_lama import run_lama_eval
from eval_chatml import run_chatml_retention_eval
from eval_passkey import run_passkey_eval
from config import DB_PATH


def clear_db():
    """Очистка результатов перед новым глобальным тестом."""
    if DB_PATH.exists():
        DB_PATH.unlink()

def run_full_suite(ablation_name, skip_layers_mask):
    """Запуск всех доступных метрик для конкретной конфигурации."""
    print(f"\n" + "="*60)
    print(f"ЗАПУСК АБЛЯЦИИ: {ablation_name} (Mask: {skip_layers_mask or 'None'})")
    print("="*60)

    #run_chatml_retention_eval(skip_layers=skip_layers_mask)
    #run_wikitext_eval(skip_layers=skip_layers_mask)
    #run_induction_eval(skip_layers=skip_layers_mask)
    #run_mcqa_eval(skip_layers=skip_layers_mask)
    #run_blimp_eval(skip_layers=skip_layers_mask)
    #run_lama_eval(skip_layers=skip_layers_mask)
    #run_passkey_eval(skip_layers=skip_layers_mask)

def get_summary_report():
    """Собирает данные из SQLite и строит финальный отчет."""
    conn = sqlite3.connect(DB_PATH)
    # Пример сборки данных из разных таблиц
    # В реальности нужно добавить колонку 'ablation_id' в каждый eval скрипт
    print("\n" + "#"*60)
    print("ФИНАЛЬНЫЙ ОТЧЕТ ПО АБЛЯЦИЯМ")
    print("#"*60)
    # Здесь можно выгрузить pandas-фреймворки и сравнить Accuracy/PPL
    conn.close()

if __name__ == "__main__":
    # Список экспериментов: (Имя, Маска)
    experiments = [
        ("Baseline", None),                # Чистая модель
        ("Skip Middle", "11-13"),          # Убираем 12-й слой (центр)
        ("Skip Deep", "20-24"),            # Убираем последние 4 слоя
        ("Skip Half", "12-24"),            # Убираем вторую половину
    ]

    clear_db()

    for name, mask in experiments:
        run_full_suite(name, mask)

    print("\n[!] Пайплайн завершен. Все данные сохранены в", DB_PATH)