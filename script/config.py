from pathlib import Path

BIN_PATH = Path("./bin/chat")
MODEL_PATH = Path("./model.gguf")
DATASETS_DIR = Path("./datasets/wikitext")
DB_PATH = Path("./script/eval_results.db")
MCQA_DATA_PATH = Path("./datasets/mcqa.jsonl")
BLIMP_DATA_PATH = Path("./datasets/blimp.jsonl")

SAFE_WORDS = [
    " apple", " house", " water", " light", " stone", " music",
    " river", " glass", " paper", " plant", " cloud", " money",
    " space", " magic", " robot", " dream", " color", " human",
    " forest", " metal", " system", " nature", " animal", " ocean"
]