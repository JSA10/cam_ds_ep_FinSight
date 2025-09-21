from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "facebook/bart-large-cnn"  # or "google/flan-t5-large"

@lru_cache(maxsize=1)
def _load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    mdl.eval()
    return tok, mdl

class ChatSeq2Seq:
    def __init__(self, max_input_tokens=2048):
        self.max_input_tokens = max_input_tokens

    @torch.inference_mode()
    def create_chat_completion(self, messages, max_tokens=700, temperature=0.2, top_p=0.9, **_):
        prompt = "\n".join([m.get("content","") for m in messages if m.get("content")])
        tok, mdl = _load()
        enc = tok(prompt, return_tensors="pt", truncation=True)
        gen = mdl.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            num_beams=4 if temperature == 0 else 1,
        )
        text = tok.decode(gen[0], skip_special_tokens=True).strip()
        return {"choices": [{"message": {"content": text}}]}
