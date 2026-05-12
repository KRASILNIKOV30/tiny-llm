import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

DB_PATH = Path("./script/eval_results.db")

# Настройки графиков
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def get_experiment_name(row):
    """Формирует понятное имя эксперимента из масок"""
    masks = []
    if row['layer_mask'] != 'None': masks.append(f"L:{row['layer_mask']}")
    if row['head_mask'] != 'None':  masks.append(f"H:{row['head_mask']}")
    if row['mlp_mask'] != 'None':   masks.append(f"MLP:{row['mlp_mask']}")
    if row['rope_mask'] != 'None':  masks.append(f"RoPE:{row['rope_mask']}")
    return " | ".join(masks) if masks else "Baseline"

def fetch_metric(table, metric_col, as_percentage=False):
    """Достает среднюю метрику из таблицы с группировкой по маскам"""
    conn = sqlite3.connect(DB_PATH)
    try:
        # Проверяем, существует ли таблица
        cursor = conn.cursor()
        cursor.execute(f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table}'")
        if cursor.fetchone()[0] == 0:
            return pd.DataFrame()

        query = f"""
            SELECT layer_mask, head_mask, mlp_mask, rope_mask, 
                   AVG({metric_col}) as {metric_col} 
            FROM {table} 
            WHERE status = 'success' OR status LIKE '%error%' -- Берем все, кроме крэшей
            GROUP BY layer_mask, head_mask, mlp_mask, rope_mask
        """
        df = pd.read_sql(query, conn)
        df['Experiment'] = df.apply(get_experiment_name, axis=1)
        if as_percentage:
            df[metric_col] = df[metric_col] * 100
        return df[['Experiment', metric_col]].set_index('Experiment')
    except Exception as e:
        print(f"[!] Ошибка чтения {table}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def build_summary_table():
    print("[*] Сборка сводной таблицы...")

    # Собираем все метрики
    dfs = [
        fetch_metric("baseline_wikitext", "perplexity"),
        fetch_metric("baseline_induction", "induction_score"),
        fetch_metric("baseline_mcqa_logits", "is_correct", True).rename(columns={"is_correct": "MCQA Acc %"}),
        fetch_metric("baseline_blimp", "is_correct", True).rename(columns={"is_correct": "BLiMP Acc %"}),
        fetch_metric("baseline_lama", "is_correct", True).rename(columns={"is_correct": "LAMA Acc %"}),
        fetch_metric("baseline_chatml_retention", "retention_score", True).rename(columns={"retention_score": "ChatML %"}),
        fetch_metric("baseline_passkey", "is_correct", True).rename(columns={"is_correct": "Passkey Acc %"})
    ]

    # Объединяем в один DataFrame
    summary_df = pd.concat([df for df in dfs if not df.empty], axis=1)
    return summary_df

def plot_bar_charts(summary_df):
    """Строит отдельные столбчатые диаграммы для каждой метрики"""
    if summary_df.empty:
        return

    # Полный список метрик
    metrics_to_plot = [
        'induction_score',
        'MCQA Acc %',
        'BLiMP Acc %',
        'LAMA Acc %',
        'ChatML %',
        'Passkey Acc %'
    ]

    for metric in metrics_to_plot:
        if metric in summary_df.columns:
            plt.figure(figsize=(10, 6))

            sns.barplot(
                x=summary_df.index,
                y=summary_df[metric],
                hue=summary_df.index,
                palette="viridis",
                legend=False
            )

            plt.title(f'Деградация: {metric}', fontsize=16)
            plt.xticks(rotation=45, ha='right')

            # Умная настройка оси Y
            if "Acc" in metric or "ChatML" in metric:
                plt.ylabel('Score (%)')
                plt.ylim(0, 105)
            elif "induction" in metric:
                plt.ylabel('Score (0.0 - 1.0)')
                plt.ylim(0, 1.1) # Induction обычно в диапазоне 0-1

            safe_name = metric.replace(' ', '_').replace('%', 'pct').lower()
            filename = f"ablation_{safe_name}.png"

            plt.tight_layout()
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"[+] Сохранен график: {filename}")

    # Perplexity оставляем особняком (шкала от 0 до бесконечности)
    if 'perplexity' in summary_df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=summary_df.index,
            y=summary_df['perplexity'],
            hue=summary_df.index,
            palette="Reds_r",
            legend=False
        )
        plt.title('Wikitext Perplexity (Меньше = Лучше)', fontsize=16)
        plt.ylabel('PPL')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('ablation_perplexity.png', dpi=300)
        plt.close()
        print("[+] Сохранен график: ablation_perplexity.png")

def plot_passkey_heatmap():
    """Строит тепловую карту для Passkey Retrieval (Baseline vs Ablation)"""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT layer_mask, mlp_mask, rope_mask, context_length, depth_pct, is_correct FROM baseline_passkey WHERE status='success'", conn)
        if df.empty: return

        # Выбираем только бейзлайн и, например, поломку RoPE
        baseline_df = df[(df['layer_mask'] == 'None') & (df['rope_mask'] == 'None')]
        rope_df = df[df['rope_mask'] == '0-24'] # Полное отключение RoPE

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, data, title in zip(axes, [baseline_df, rope_df], ["Baseline", "No RoPE (0-24)"]):
            if data.empty: continue
            # Сводная таблица: Строки = Глубина, Колонки = Контекст, Значения = Успех (в %)
            pivot = data.pivot_table(index='depth_pct', columns='context_length', values='is_correct', aggfunc='mean') * 100
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=100, ax=ax)
            ax.set_title(f'Passkey Retrieval: {title}', fontsize=14)
            ax.set_ylabel('Глубина иголки (Depth %)')
            ax.set_xlabel('Размер контекста (Слов)')

        plt.tight_layout()
        plt.savefig('passkey_heatmap.png', dpi=300)
        print("[+] Сохранен график: passkey_heatmap.png")

    except Exception as e:
        print(f"[-] Не удалось построить Heatmap для Passkey: {e}")
    finally:
        conn.close()

def export_for_ai(summary_df):
    """Экспортирует данные в чистый JSON для анализа в LLM"""
    if summary_df.empty:
        return

    # Преобразуем индексы (имена экспериментов) в колонку
    export_data = summary_df.reset_index().to_dict(orient='records')

    # Очищаем NaN значения (если какой-то тест упал/не запускался)
    cleaned_data = []
    for row in export_data:
        clean_row = {
            k: (round(v, 2) if isinstance(v, (int, float)) and pd.notnull(v) else v)
            for k, v in row.items()
        }
        clean_row = {k: (v if pd.notnull(v) else "N/A") for k, v in clean_row.items()}
        cleaned_data.append(clean_row)

    output_json = {
        "context": "Это результаты абляционного анализа LLM модели Qwen2.5-0.5B.",
        "metrics_description": {
            "perplexity": "Способность предсказывать текст (Меньше = лучше).",
            "induction_score": "Способность копировать предыдущие паттерны (Больше = лучше).",
            "MCQA Acc %": "Логика и фактология в multiple-choice (Больше = лучше).",
            "BLiMP Acc %": "Понимание синтаксиса и грамматики (Больше = лучше).",
            "LAMA Acc %": "Завершение фактов (Больше = лучше).",
            "ChatML %": "Способность удерживать системный формат ChatML (Больше = лучше).",
            "Passkey Acc %": "Способность находить информацию в длинном контексте (Больше = лучше)."
        },
        "results": cleaned_data
    }

    with open('ablation_summary_for_ai.json', 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    print("[+] Сохранен JSON для ИИ: ablation_summary_for_ai.json")

if __name__ == "__main__":
    df_summary = build_summary_table()

    if not df_summary.empty:
        print("\n=== СВОДНАЯ ТАБЛИЦА МЕТРИК ===")
        print(df_summary.round(2).to_string())

        plot_bar_charts(df_summary)
        plot_passkey_heatmap()
        export_for_ai(df_summary)
    else:
        print("[!] Нет данных для анализа. Запусти main.py с if_exists='append' в eval скриптах.")