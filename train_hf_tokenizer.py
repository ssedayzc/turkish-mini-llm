"""Train a small BPE tokenizer and upload it to Hugging Face.

Install once:
    pip install transformers tokenizers huggingface_hub
    hf auth login

Then edit TEXT_FILE and REPO_ID below and run:
    python train_hf_tokenizer.py
"""

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

TEXT_FILE = "text.txt"
REPO_ID = "sedayzc/turkish-tokenizer"
VOCAB_SIZE = 512

from pathlib import Path

print("Çalışan dosya:", Path(__file__).resolve())
print("Repo ID:", REPO_ID)

SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]


# 1. Create and train a byte-level BPE tokenizer.
backend = Tokenizer(BPE(unk_token="<unk>"))
backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
backend.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=ByteLevel.alphabet(),
)
backend.train([TEXT_FILE], trainer)

# 2. Wrap it so Transformers and AutoTokenizer understand it.
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=backend,
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<bos>",
    eos_token="<eos>",
)

# 3. Save locally and upload to the Hub.
tokenizer.save_pretrained("my-tokenizer")
tokenizer.push_to_hub(REPO_ID)

# 4. Load it exactly as other users will load it.
loaded = AutoTokenizer.from_pretrained(REPO_ID)
example = "Bu bilmecenin anlamını çözmek günler sürdü"

token_ids = loaded.encode(example)
print("Tokens:", loaded.convert_ids_to_tokens(token_ids))
print("Token IDs:", token_ids)
print("Decoded:", loaded.decode(token_ids))
