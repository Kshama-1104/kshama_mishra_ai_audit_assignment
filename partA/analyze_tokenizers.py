from pathlib import Path
import regex
import tiktoken
from transformers import AutoTokenizer

CORPUS_DIR = Path(__file__).parent / "corpus_flores200"
FILES = {
    "eng": "eng.txt",
    "hin": "hin.txt",
    "kan": "kan.txt",
    "tam": "tam.txt",
    "tel": "tel.txt",
}

def main():
    gpt2 = tiktoken.get_encoding("gpt2")
    xlm = AutoTokenizer.from_pretrained("xlm-roberta-base")

    tokenizers = {
        "gpt2": lambda text: gpt2.encode(text),
        "xlm-roberta-base": lambda text: xlm.encode(text, add_special_tokens=False),
    }

    print("tokenizer,lang,sentences,tokens_per_sentence,tokens_per_word,tokens_per_grapheme,tokens_per_utf8_byte")

    for tok_name, encode in tokenizers.items():
        for lang, filename in FILES.items():
            lines = (CORPUS_DIR / filename).read_text(encoding="utf-8").splitlines()
            total_tokens = sum(len(encode(line)) for line in lines)
            total_words = sum(len(line.split()) for line in lines)
            total_graphemes = sum(len(regex.findall(r"\X", line)) for line in lines)
            total_bytes = sum(len(line.encode("utf-8")) for line in lines)

            print(
                tok_name,
                lang,
                len(lines),
                round(total_tokens / len(lines), 3),
                round(total_tokens / total_words, 3),
                round(total_tokens / total_graphemes, 4),
                round(total_tokens / total_bytes, 4),
                sep=",",
            )

if __name__ == "__main__":
    main()
