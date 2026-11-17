import pytest, torch


def make_tiny(vocab_extra="", layers=4, seed=0):
    from tokenizers import Tokenizer, Regex
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    chars = [chr(c) for c in range(32, 127)] + ["\n"]
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    for c in chars:
        vocab[c] = len(vocab)
    tk = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = Split(Regex("[\\s\\S]"), behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}assistant: {% endif %}")
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      eos_token_id=1, pad_token_id=0, tie_word_embeddings=False)
    model = Qwen2ForCausalLM(cfg).eval()
    return model, tok


@pytest.fixture(scope="session")
def tiny():
    return make_tiny()
