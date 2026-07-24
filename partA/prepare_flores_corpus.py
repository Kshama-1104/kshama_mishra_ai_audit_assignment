from datasets import load_dataset
from pathlib import Path

LANGS = {
    "eng_Latn": "eng",
    "hin_Deva": "hin",
    "kan_Knda": "kan",
    "tam_Taml": "tam",
    "tel_Telu": "tel",
}

def main():
    out = Path(__file__).parent / "corpus_flores200"
    out.mkdir(exist_ok=True)

    for config, short in LANGS.items():
        ds = load_dataset("Muennighoff/flores200", config, split="dev", trust_remote_code=True)
        lines = [row["sentence"].strip() for row in ds if row.get("sentence", "").strip()]
        (out / f"{short}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(short, config, len(lines), lines[0][:80])

if __name__ == "__main__":
    main()
