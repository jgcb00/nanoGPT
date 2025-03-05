print("go")

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

print("done importing")

class GPT_LM(LM):
    def __init__(self):
        super().__init__()

    def loglikelihood(self, requests):
        return

    def generate_until(self, requests: list[Instance]) -> list[str]:
        return
    
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        return
    
lm = GPT_LM()
_ = lm_eval.simple_evaluate(lm, tasks=["swde"])
