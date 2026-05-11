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
    if DB_PATH.exists():
        DB_PATH.unlink()

def run_full_suite(ablation_name, skip_layers_mask=None, head_mask=None):
    print(f"\n" + "="*60)
    print(f"ЗАПУСК АБЛЯЦИИ: {ablation_name} | Layers: {skip_layers_mask or 'All'} | Heads: {head_mask or 'All'}")
    print("="*60)

    # Передаем head_mask во все функции (раскомментируй нужные)
    run_induction_eval(skip_layers=skip_layers_mask, head_mask=head_mask)
    #run_wikitext_eval(skip_layers=skip_layers_mask, head_mask=head_mask)
    # run_chatml_retention_eval(skip_layers=skip_layers_mask, head_mask=head_mask)
    # run_mcqa_eval(skip_layers=skip_layers_mask, head_mask=head_mask)
    # run_blimp_eval(skip_layers=skip_layers_mask, head_mask=head_mask)
    # run_lama_eval(skip_layers=skip_layers_mask, head_mask=head_mask)
    # run_passkey_eval(skip_layers=skip_layers_mask, head_mask=head_mask)

# ... (get_summary_report остается без изменений) ...

if __name__ == "__main__":
    experiments = [
        ("Baseline", None, None),
        #("Skip Middle", "11-13"),
        #("Skip Deep", "20-24"),
        #("Skip Half", "12-24"),
        ("Zero Q-Head L12:Q:0", None, "12:q:0"),
        #("Zero Q-Head L12:Q:5", None, "12:q:5"),
        ("Zero KV-Head L12:KV:0", None, "12:kv:0"),
        #("Zero KV-Head L12:KV:1", None, "12:kv:1"),
    ]

    clear_db()

    for name, l_mask, h_mask in experiments:
        run_full_suite(name, l_mask, h_mask)

    print("\n[!] Пайплайн завершен. Все данные сохранены в", DB_PATH)