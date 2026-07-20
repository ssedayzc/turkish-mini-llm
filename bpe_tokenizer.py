"""
Turkish Byte-Level BPE Tokenizer

Bu script küçük ölçekli bir Türkçe Byte-Level BPE tokenizer eğitir,
lokal olarak kaydeder ve örnek bir metin üzerinde tokenizer çıktısını
gösterir.

Kurulum:
    pip install transformers tokenizers

Çalıştırma:
    python bpe_tokenizer.py
"""

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


# ============================================================
# AYARLAR
# ============================================================

TEXT_FILE = "text.txt"

OUTPUT_DIR = "my-tokenizer"

VOCAB_SIZE = 512

MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "<unk>",
    "<pad>",
    "<bos>",
    "<eos>",
]


# ============================================================
# DOSYA KONTROLÜ
# ============================================================

text_path = Path(
    TEXT_FILE
)

if not text_path.exists():

    raise FileNotFoundError(
        f"Eğitim metni bulunamadı: "
        f"{text_path.resolve()}"
    )


print(
    "=" * 70
)

print(
    "TURKISH BYTE-LEVEL BPE TOKENIZER"
)

print(
    "=" * 70
)

print(
    "Eğitim dosyası:",
    text_path.resolve(),
)

print(
    "Vocabulary size:",
    VOCAB_SIZE,
)

print(
    "Minimum frequency:",
    MIN_FREQUENCY,
)

print(
    "Special tokens:",
    SPECIAL_TOKENS,
)


# ============================================================
# 1. BPE TOKENIZER BACKEND
# ============================================================

backend = Tokenizer(
    BPE(
        unk_token="<unk>"
    )
)


# ============================================================
# 2. BYTE-LEVEL PRE-TOKENIZER
# ============================================================

backend.pre_tokenizer = (
    ByteLevel(
        add_prefix_space=False
    )
)


# ============================================================
# 3. BYTE-LEVEL DECODER
# ============================================================

backend.decoder = (
    ByteLevelDecoder()
)


# ============================================================
# 4. BPE TRAINER
# ============================================================

trainer = BpeTrainer(

    vocab_size=VOCAB_SIZE,

    min_frequency=MIN_FREQUENCY,

    special_tokens=(
        SPECIAL_TOKENS
    ),

    initial_alphabet=(
        ByteLevel.alphabet()
    ),
)


# ============================================================
# 5. TOKENIZER'I EĞİT
# ============================================================

print(
    "\nTokenizer eğitiliyor..."
)

backend.train(
    [
        TEXT_FILE
    ],
    trainer,
)

print(
    "Tokenizer eğitimi tamamlandı."
)


# ============================================================
# 6. TRANSFORMERS UYUMLU TOKENIZER
# ============================================================

tokenizer = (
    PreTrainedTokenizerFast(

        tokenizer_object=(
            backend
        ),

        unk_token="<unk>",

        pad_token="<pad>",

        bos_token="<bos>",

        eos_token="<eos>",
    )
)


# ============================================================
# 7. LOKAL KAYIT
# ============================================================

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print(
    "\nTokenizer kaydedildi:"
)

print(
    Path(
        OUTPUT_DIR
    ).resolve()
)


# ============================================================
# 8. TOKENIZER BİLGİLERİ
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TOKENIZER BİLGİLERİ"
)

print(
    "=" * 70
)

print(
    "Vocabulary size:",
    len(tokenizer),
)

print(
    "UNK token:",
    tokenizer.unk_token,
)

print(
    "PAD token:",
    tokenizer.pad_token,
)

print(
    "BOS token:",
    tokenizer.bos_token,
)

print(
    "EOS token:",
    tokenizer.eos_token,
)


# ============================================================
# 9. ÖRNEK TOKENIZATION
# ============================================================

example = (
    "Bu bilmecenin anlamını çözmek "
    "günler sürdü"
)

print(
    "\n"
    + "=" * 70
)

print(
    "ÖRNEK TOKENIZATION"
)

print(
    "=" * 70
)

print(
    "Metin:"
)

print(
    example
)


token_ids = tokenizer.encode(
    example
)

tokens = (
    tokenizer
    .convert_ids_to_tokens(
        token_ids
    )
)

decoded = tokenizer.decode(
    token_ids
)


print(
    "\nTokens:"
)

print(
    tokens
)

print(
    "\nToken IDs:"
)

print(
    token_ids
)

print(
    "\nDecoded:"
)

print(
    decoded
)


# ============================================================
# 10. TOKEN SAYISI
# ============================================================

print(
    "\nToken sayısı:",
    len(token_ids),
)


# ============================================================
# SONUÇ
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TOKENIZER HAZIR"
)

print(
    "=" * 70
)

print(
    "\nHugging Face:"
)

print(
    "sedayzc/turkish-bpe-tokenizer"
)