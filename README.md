---

# 🔤 Turkish Byte-Level BPE Tokenizer

Bu repository kapsamında Türkçe metinler üzerinde küçük ölçekli bir
**Byte-Level BPE (Byte Pair Encoding) Tokenizer** sıfırdan eğitilmiştir.

Tokenizer, Hugging Face `tokenizers` kütüphanesi kullanılarak oluşturulmuş
ve `transformers` kütüphanesi ile uyumlu hale getirilmiştir.

Tokenizer Hugging Face Hub üzerinde yayınlanmaktadır:

[![Hugging Face Tokenizer](https://img.shields.io/badge/🤗%20Hugging%20Face-Turkish%20BPE%20Tokenizer-FFD21E?style=for-the-badge)](https://huggingface.co/sedayzc/turkish-bpe-tokenizer)

```text
sedayzc/turkish-bpe-tokenizer
```

---

## ⚙️ Tokenizer Yapılandırması

| Özellik | Değer |
|---|---|
| Tokenizer | Byte-Level BPE |
| Dil | Türkçe |
| Vocabulary Size | 512 |
| Minimum Frequency | 2 |
| Pre-tokenizer | ByteLevel |
| Decoder | ByteLevel |

Tokenizer aşağıdaki özel tokenları kullanmaktadır:

| Token | Açıklama |
|---|---|
| `<unk>` | Bilinmeyen token |
| `<pad>` | Padding token |
| `<bos>` | Sequence başlangıcı |
| `<eos>` | Sequence sonu |

---

## 🧩 Tokenizer Pipeline

```mermaid
flowchart LR

    A["📝 Türkçe Eğitim Metni"]
    B["🔡 Byte-Level Pre-Tokenizer"]
    C["🔗 BPE Pair Merging"]
    D["📚 512 Token Vocabulary"]
    E["🤗 PreTrainedTokenizerFast"]
    F["☁️ Hugging Face Hub"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

---

## 🚀 Kullanım

Tokenizer Hugging Face üzerinden doğrudan yüklenebilir:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "sedayzc/turkish-bpe-tokenizer"
)

text = (
    "Bu bilmecenin anlamını "
    "çözmek günler sürdü"
)

token_ids = tokenizer.encode(
    text
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
    "Tokens:",
    tokens
)

print(
    "Token IDs:",
    token_ids
)

print(
    "Decoded:",
    decoded
)
```

---

## 📊 Örnek Çıktı

Örnek girdi:

```text
Bu bilmecenin anlamını çözmek günler sürdü
```

Örnek tokenizer çıktısı:

```text
Tokens:
[
    'B',
    'u',
    'Ġb',
    'i',
    'lm',
    'e',
    'c',
    'en',
    'in',
    ...
]
```

Token ID çıktısı:

```text
[
    37,
    88,
    268,
    76,
    313,
    ...
]
```

Decode sonucu:

```text
Bu bilmecenin anlamını çözmek günler sürdü
```

Byte-Level BPE gösteriminde görülebilen `Ġ` gibi semboller,
tokenizer'ın byte-level iç temsilinden kaynaklanmaktadır.
Decode işlemi sonrasında metin normal Türkçe biçimine geri dönmektedir.

---

## 📁 Tokenizer Dosyaları

Tokenizer eğitildikten sonra aşağıdaki dosyalar oluşturulmaktadır:

```text
my-tokenizer/
│
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

Bu dosyalar tokenizer'ın Hugging Face Transformers ile yüklenebilmesini
sağlamaktadır.

---

## 🧪 Lokal Eğitim

Tokenizer'ı lokal olarak yeniden eğitmek için:

```bash
python bpe_tokenizer.py
```

Script:

```text
text.txt
```

dosyasını kullanarak tokenizer'ı eğitir ve çıktıları:

```text
my-tokenizer/
```

klasörüne kaydeder.

---

## ⚠️ Sınırlamalar

Bu tokenizer küçük ölçekli bir Türkçe eğitim metni üzerinde ve
`512` vocabulary size ile eğitilmiştir.

Bu nedenle büyük üretim modellerinde kullanılan tokenizer'larla aynı
kapsam ve token verimliliğine sahip olması beklenmemelidir.

Tokenizer'ın temel amacı:

- BPE algoritmasını deneyimlemek
- Tokenizer eğitim sürecini anlamak
- Küçük LLM deneylerinde kullanılmak
- Hugging Face tokenizer entegrasyonunu göstermek

olarak belirlenmiştir.