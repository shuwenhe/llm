import json


class CharTokenizer:
    def __init__(self, stoi=None, itos=None):
        self.stoi = stoi or {}
        self.itos = itos or {}

    @classmethod
    def from_texts(cls, texts):
        chars = sorted(set("".join(texts)))
        stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        stoi["<unk>"] = 0
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text):
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids)

    @property
    def vocab_size(self):
        return max(self.stoi.values()) + 1 if self.stoi else 1

    def to_dict(self):
        return {"stoi": self.stoi}

    @classmethod
    def from_dict(cls, d):
        stoi = d.get("stoi", {})
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)
