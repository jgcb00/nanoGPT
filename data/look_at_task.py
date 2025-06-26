#!/usr/bin/env python3

import json
import random
import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
import tiktoken
import pickle

CAPTURED = []


class GPT_LM(LM):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def loglikelihood(self, requests):
        # requests: list of Instance(task_name, args=(input_str, target_str), kwargs={})
        # capture all incoming args
        for inst in requests:
            inp, tgt = inst.args
            length = len(self.tokenizer.encode(inp))
            CAPTURED.append(
                {"task": inst.task_name, "input": inp, "target": tgt, "length": length}
            )
        # return dummy scores
        return [(0.0, False) for _ in requests]

    def generate_until(self, requests: list[Instance]) -> list[str]:
        # capture every (input, kwargs) for generation tasks
        for inst in requests:
            inp, gen_kwargs = inst.args
            length = len(self.tokenizer.encode(inp))
            CAPTURED.append(
                {
                    "task": inst.task_name,
                    "input": inp,
                    "kwargs": gen_kwargs,
                    "length": length,
                }
            )
        # dummy outputs so lm_eval keeps going
        return [""] * len(requests)

    def loglikelihood_rolling(self, requests):
        # stub out
        return [(0.0, False) for _ in requests]


def main():
    task = "ruler_qa_squad"

    # load tokenizer
    with open("data/enc.pkl", "rb") as f:
        enc_pickled = pickle.load(f)
    enc = tiktoken.core.Encoding(enc_pickled.pop("name"), **enc_pickled)

    lm = GPT_LM(enc)
    lm_eval.simple_evaluate(
        lm,
        tasks=[task],
        limit=20,
        model_args={"tokenizer": "gpt2"},
        metadata={"max_seq_lengths": [8192]},
    )

    sample = random.sample(CAPTURED, min(len(CAPTURED), 5))
    with open(f"samples_{task}.json", "w") as f:
        json.dump(sample, f, indent=2)
    print(f"Wrote {len(sample)} examples to samples_{task}.json")


if __name__ == "__main__":
    main()
