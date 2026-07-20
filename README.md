
# Turkish Byte-Level BPE Tokenizer

Bu repository kapsamında Türkçe metinler üzerinde küçük ölçekli bir
**Byte-Level BPE (Byte Pair Encoding) Tokenizer** sıfırdan eğitilmiştir.

Tokenizer, Hugging Face `tokenizers` kütüphanesi kullanılarak oluşturulmuş
ve `transformers` kütüphanesi ile uyumlu hale getirilmiştir.

Tokenizer Hugging Face Hub üzerinde yayınlanmaktadır:

[![Hugging Face Tokenizer](https://img.shields.io/badge/Hugging%20Face-Turkish%20BPE%20Tokenizer-FFD21E)](https://huggingface.co/sedayzc/turkish-bpe-tokenizer)

`sedayzc/turkish-bpe-tokenizer`

## ⚙️ Tokenizer Yapılandırması

| Özellik | Değer |
|---|---|
| Tokenizer | Byte-Level BPE |
| Dil | Türkçe |
| Vocabulary Size | 512 |
| Minimum Frequency | 2 |
| Pre-tokenizer | ByteLevel |
| Decoder | ByteLevel |

## Özel Tokenlar

| Token | Açıklama |
|---|---|
| `<unk>` | Bilinmeyen token |
| `<pad>` | Padding token |
| `<bos>` | Sequence başlangıcı |
| `<eos>` | Sequence sonu |

## 🚀 Kullanım

Tokenizer Hugging Face üzerinden doğrudan yüklenebilir:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "sedayzc/turkish-bpe-tokenizer"
)

text = "Bu bilmecenin anlamını çözmek günler sürdü"

token_ids = tokenizer.encode(text)
tokens = tokenizer.convert_ids_to_tokens(token_ids)
decoded = tokenizer.decode(token_ids)

print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Decoded:", decoded)
