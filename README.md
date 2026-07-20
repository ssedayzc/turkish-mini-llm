<div align="center">

# 🔤 Turkish Byte-Level BPE Tokenizer

### A Byte-Level BPE Tokenizer Trained on Turkish Text

Türkçe metinler üzerinde sıfırdan eğitilmiş küçük ölçekli  
**Byte-Level BPE (Byte Pair Encoding) Tokenizer**

<br>

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Turkish%20BPE%20Tokenizer-FFD21E?style=for-the-badge)](https://huggingface.co/sedayzc/turkish-bpe-tokenizer)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-Compatible-FFD21E?style=for-the-badge)](https://huggingface.co/docs/transformers/)
[![BPE](https://img.shields.io/badge/Tokenizer-Byte--Level%20BPE-blue?style=for-the-badge)](https://huggingface.co/sedayzc/turkish-bpe-tokenizer)

<br>

**Byte-Level BPE** • **512 Vocabulary Size** • **Turkish** • **Hugging Face Compatible**

</div>

---

## 📌 Proje Hakkında

Bu proje, Türkçe metinler üzerinde **Byte-Level BPE (Byte Pair Encoding)**
tokenizer'ın sıfırdan eğitilmesini ve Hugging Face ekosistemiyle uyumlu
hale getirilmesini içermektedir.

Tokenizer, Hugging Face `tokenizers` kütüphanesi kullanılarak eğitilmiş ve
`PreTrainedTokenizerFast` aracılığıyla `transformers` kütüphanesiyle uyumlu
hale getirilmiştir.

Eğitilmiş tokenizer Hugging Face Hub üzerinde yayınlanmaktadır:

[![Open on Hugging Face](https://img.shields.io/badge/🤗%20Open%20on%20Hugging%20Face-Turkish%20BPE%20Tokenizer-FFD21E?style=for-the-badge)](https://huggingface.co/sedayzc/turkish-bpe-tokenizer)

---

## 🎯 Amaç

Bu projenin temel amacı, Türkçe metinler için özel bir tokenizer'ın
sıfırdan nasıl eğitilebileceğini deneyimlemek ve Byte Pair Encoding
algoritmasının çalışma mekanizmasını uygulamalı olarak incelemektir.

Proje kapsamında:

- Türkçe metinlerden oluşan bir eğitim corpus'u hazırlanmıştır.
- Byte-Level tokenization yaklaşımı kullanılmıştır.
- BPE algoritması ile tokenizer sıfırdan eğitilmiştir.
- 512 tokenlık bir vocabulary oluşturulmuştur.
- Özel tokenlar tanımlanmıştır.
- Tokenizer, Hugging Face Transformers ile uyumlu hale getirilmiştir.
- Eğitilmiş tokenizer Hugging Face Hub üzerinde yayınlanmıştır.

---

## 📂 Repository Yapısı

```text
turkish-mini-llm/
│
├── data/
│
├── deepseek3/
│
├── gemma4/
│
├── my-tokenizer/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── qwen3/
│
├── bpe_tokenizer.py
├── demo.ipynb
├── text.txt
├── README.md
├── .gitignore
└── .python-version
```

`bpe_tokenizer.py`, Byte-Level BPE tokenizer'ın eğitim sürecini içermektedir.

`my-tokenizer/` klasörü ise eğitim sonucunda oluşturulan tokenizer
dosyalarını barındırmaktadır.

`demo.ipynb`, tokenizer ve proje kapsamında gerçekleştirilen deneyler için
kullanılan notebook dosyasıdır.

---

## ⚙️ Tokenizer Yapılandırması

| Özellik | Değer |
|---|---|
| Tokenizer Algoritması | Byte-Level BPE |
| Dil | Türkçe 🇹🇷 |
| Vocabulary Size | 512 |
| Minimum Frequency | 2 |
| Pre-tokenizer | ByteLevel |
| Decoder | ByteLevel |
| Framework | Hugging Face Tokenizers |
| Transformers Uyumluluğu | Evet |
| AutoTokenizer Desteği | Evet |

---

## 🔑 Özel Tokenlar

Tokenizer aşağıdaki özel tokenları kullanmaktadır:

| Token | Açıklama |
|---|---|
| `<unk>` | Bilinmeyen token |
| `<pad>` | Padding token |
| `<bos>` | Sequence başlangıcı |
| `<eos>` | Sequence sonu |

---

## 🧠 Byte-Level BPE Nedir?

**Byte Pair Encoding (BPE)**, metin içerisindeki sık tekrar eden sembol veya
token çiftlerini iteratif olarak birleştiren bir subword tokenization
algoritmasıdır.

Örneğin sık tekrar eden:

```text
t + ü → tü
```

ve daha sonra:

```text
tü + rk → türk
```

gibi birleşimler öğrenilebilir.

Byte-Level BPE yaklaşımında işlem karakterlerden ziyade byte düzeyinde
gerçekleştirildiği için tokenizer çok geniş bir karakter kümesini
destekleyebilir.

Bu yaklaşım özellikle bilinmeyen karakterlerin ve farklı Unicode
sembollerinin işlenebilmesi açısından avantaj sağlar.

---

## 🚀 Hugging Face Üzerinden Kullanım

Tokenizer, Hugging Face Hub üzerinden doğrudan yüklenebilir:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "sedayzc/turkish-bpe-tokenizer"
)
```

Bir Türkçe metni tokenize etmek için:

```python
text = "Bu bilmecenin anlamını çözmek günler sürdü"

tokens = tokenizer.tokenize(text)

print(tokens)
```

Token ID'lerini elde etmek için:

```python
token_ids = tokenizer.encode(text)

print(token_ids)
```

Token ID'lerini tekrar metne dönüştürmek için:

```python
decoded = tokenizer.decode(token_ids)

print(decoded)
```

---

## 🧪 Tokenization Örneği

Örnek metin:

```text
Bu bilmecenin anlamını çözmek günler sürdü
```

Tokenizer kullanılarak:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "sedayzc/turkish-bpe-tokenizer"
)

text = "Bu bilmecenin anlamını çözmek günler sürdü"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)
decoded = tokenizer.decode(token_ids)

print("Original Text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)

print("\nDecoded Text:")
print(decoded)
```

Byte-Level BPE tokenlarında `Ġ` gibi özel gösterimler görülebilir. Bunlar
tokenizer'ın byte-level iç temsilinden kaynaklanmaktadır ve orijinal metnin
bir parçası değildir.

Decode işlemi sonrasında metin tekrar normal biçimine dönüştürülür.

---

## 💻Kullanım


```bash
git clone https://github.com/ssedayzc/turkish-mini-llm.git
```

Gerekli kütüphaneleri yükleyin:

```bash
pip install tokenizers transformers huggingface-hub
```

Tokenizer eğitim scriptini çalıştırmak için:

```bash
python bpe_tokenizer.py
```

---

## 📦 Eğitilmiş Tokenizer Dosyaları

Eğitim sonucunda oluşturulan tokenizer dosyaları `my-tokenizer/`
klasöründe bulunmaktadır:

```text
my-tokenizer/
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

Bu dosyalar tokenizer'ın Hugging Face Transformers ile yüklenebilmesini ve
tekrar kullanılabilmesini sağlar.

---

## 🤗 Hugging Face

Eğitilmiş tokenizer Hugging Face Hub üzerinde yayınlanmaktadır:

### `sedayzc/turkish-bpe-tokenizer`

[![Hugging Face Tokenizer](https://img.shields.io/badge/🤗%20View%20Tokenizer%20on%20Hugging%20Face-FFD21E?style=for-the-badge)](https://huggingface.co/sedayzc/turkish-bpe-tokenizer)

Tokenizer'ı tek satırda yükleyebilirsiniz:

```python
tokenizer = AutoTokenizer.from_pretrained(
    "sedayzc/turkish-bpe-tokenizer"
)
```

---

## ⚠️ Sınırlamalar

Bu tokenizer küçük ölçekli bir Türkçe eğitim corpus'u üzerinde ve
`512` vocabulary size ile eğitilmiştir.

Bu nedenle:

- Büyük ölçekli production modellerindeki tokenizer'larla aynı token
  verimliliğine sahip olması beklenmemelidir.
- Morfolojik olarak karmaşık Türkçe kelimeler birden fazla subword tokenına
  ayrılabilir.
- Genel amaçlı büyük Türkçe dil modelleri için optimize edilmemiştir.
- Eğitim ve deneysel kullanım amacıyla geliştirilmiştir.

<div align="center">

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-sedayzc-FFD21E?style=for-the-badge)](https://huggingface.co/sedayzc)


</div>