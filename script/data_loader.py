from config import DATASETS_DIR

def load_local_dataset():
    """
    Считывает все .txt файлы из локальной директории датасета.
    Возвращает список словарей с именем файла и его содержимым.
    """
    if not DATASETS_DIR.exists():
        raise FileNotFoundError(f"Папка с датасетами не найдена: {DATASETS_DIR}. Создайте её и добавьте .txt файлы.")

    chunks = []
    for file_path in sorted(DATASETS_DIR.glob("*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if len(text) > 100:
                chunks.append({"filename": file_path.name, "text": text})

    print(f"[*] Загружено {len(chunks)} локальных файлов из {DATASETS_DIR}")
    return chunks