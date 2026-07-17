"""Download the tiny-shakespeare corpus (public domain, ~1.1 MB).

This is the canonical character-level language-modeling dataset,
mirrored from Andrej Karpathy's char-rnn repository.

Run:  python data/download_data.py
"""

from pathlib import Path
from urllib.request import urlopen

URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
       "master/data/tinyshakespeare/input.txt")
OUT = Path(__file__).resolve().parent / "corpus.txt"


def main():
    print(f"downloading {URL} ...")
    text = urlopen(URL, timeout=30).read().decode("utf-8")
    OUT.write_text(text, encoding="utf-8")
    print(f"saved {len(text):,} characters -> {OUT}")


if __name__ == "__main__":
    main()
