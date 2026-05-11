import sqlite3
import time # <-- Не забудь импортировать time
from eval_wikitext import run_wikitext_eval
from eval_induction import run_induction_eval
from eval_mcqa import run_mcqa_eval
from eval_blimp import run_blimp_eval
from eval_lama import run_lama_eval
from eval_chatml import run_chatml_retention_eval
from eval_passkey import run_passkey_eval
from config import DB_PATH

def clear_db():
    if DB_PATH.exists():
        DB_PATH.unlink()

def run_full_suite(ablation_name, skip_layers=None, head_mask=None, mlp_mask=None, rope_mask=None):
    print(f"\n" + "="*60)
    print(f"ЗАПУСК АБЛЯЦИИ: {ablation_name}")
    print(f"Маски -> Layers: {skip_layers}, Heads: {head_mask}, MLP: {mlp_mask}, RoPE: {rope_mask}")
    print("="*60)

    suite_start_time = time.time()

    # Вспомогательная функция для запуска и замера времени каждого этапа
    def run_and_measure(test_name, eval_func, **kwargs):
        print(f"\n---> Старт этапа: {test_name} ...")
        step_start_time = time.time()

        # Вызов самой функции оценки
        eval_func(**kwargs)

        step_time = time.time() - step_start_time
        print(f"---> Завершено: {test_name} | Время этапа: {step_time:.2f} сек ({step_time/60:.2f} мин)")

    # Раскомментируй нужные тесты
    run_and_measure("Induction Heads", run_induction_eval, skip_layers=skip_layers, head_mask=head_mask, mlp_mask=mlp_mask, rope_mask=rope_mask)
    run_and_measure("Wikitext PPL", run_wikitext_eval, skip_layers=skip_layers, head_mask=head_mask, mlp_mask=mlp_mask, rope_mask=rope_mask)
    run_and_measure("ChatML Retention", run_chatml_retention_eval, skip_layers=skip_layers, head_mask=head_mask, mlp_mask=mlp_mask, rope_mask=rope_mask)
    run_and_measure("MCQA", run_mcqa_eval, skip_layers=skip_layers, head_mask=head_mask, mlp_mask=mlp_mask, rope_mask=rope_mask)
    run_and_measure("BLiMP", run_blimp_eval, skip_layers=skip_layers, head_mask=head_mask, mlp_mask=mlp_mask, rope_mask=rope_mask)
    run_and_measure("LAMA", run_lama_eval, skip_layers=skip_layers, head_mask=head_mask, mlp_mask=mlp_mask, rope_mask=rope_mask)
    run_and_measure("Passkey", run_passkey_eval, skip_layers=skip_layers, head_mask=head_mask, mlp_mask=mlp_mask, rope_mask=rope_mask)

    suite_time = time.time() - suite_start_time
    print(f"\n[!] Абляция '{ablation_name}' полностью завершена за {suite_time/60:.2f} минут.")


if __name__ == "__main__":
    experiments = [
        # (Имя, Маска Слоев, Маска Голов, Маска MLP, Маска RoPE)
        ("Baseline", None, None, None, None),

        # 1. Абляции целых слоев
        ("Skip Early (2-4)", "2-4", None, None, None),
        ("Skip Middle (11-13)", "11-13", None, None, None),
        ("Skip Deep (21-23)", "21-23", None, None, None),

        # 2. Абляции голов внимания
        ("Zero Q-Head L12:Q:5", None, "12:q:5", None, None),
        ("Zero KV-Head L12:KV:0", None, "12:kv:0", None, None),

        # 3. Эксперименты с MLP
        ("Sever MLP Early (4-6)", None, None, "4-6", None),
        ("Sever MLP Deep (20-22)", None, None, "20-22", None),

        # 4. Эксперименты с RoPE
        ("Mutilate RoPE Early (0-3)", None, None, None, "0-3"),
        ("Mutilate RoPE Deep (20-23)", None, None, None, "20-23"),
        ("Mutilate RoPE All Layers", None, None, None, "0-24"),
    ]

    clear_db()

    global_start_time = time.time()

    for idx, (name, l_mask, h_mask, m_mask, r_mask) in enumerate(experiments):
        print(f"\n\n>>> Прогресс пайплайна: Эксперимент {idx+1} из {len(experiments)} <<<")
        run_full_suite(name, l_mask, h_mask, m_mask, r_mask)

    global_time = time.time() - global_start_time

    # Итоговый лог с конвертацией в часы и минуты
    hours = int(global_time // 3600)
    minutes = int((global_time % 3600) // 60)

    print("\n" + "#"*60)
    print(f"[!] ГЛОБАЛЬНЫЙ ПАЙПЛАЙН ЗАВЕРШЕН!")
    print(f"[!] Общее затраченное время: {hours} ч. {minutes} мин. ({global_time:.2f} сек.)")
    print(f"[!] Все данные успешно сохранены в: {DB_PATH}") [cite: 93-94]
    print("#"*60 + "\n")