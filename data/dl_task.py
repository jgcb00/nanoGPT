print("go")

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

print("done importing")


class GPT_LM(LM):
    def __init__(self):
        super().__init__()

    def loglikelihood(self, requests):
        return [(0.0, False) for _ in requests]

    def generate_until(self, requests: list[Instance]) -> list[str]:
        return [""] * len(requests)

    def loglikelihood_rolling(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
        print("not implemented")
        return


# tasks we care about: hellaswag, swde, squadv2, squad_completion, fda, nq_open, drop
# + : mmlu, triviaqa, arc, piqa, winogrande

# hendrycks_math, gsm8k

lm = GPT_LM()
# _ = lm_eval.simple_evaluate(lm, tasks=["triviaqa"])
# print("triviaqa done")
# _ = lm_eval.simple_evaluate(lm, tasks=["truthfulqa_mc1"])
# _ = lm_eval.simple_evaluate(lm, tasks=["truthfulqa_mc2"])
# print("truthfulqa done")
# _ = lm_eval.simple_evaluate(lm, tasks=["belebele"])
# print("belebele done")
# import os
# os.environ["HF_ALLOW_CODE_EVAL"] = "1"
# _ = lm_eval.simple_evaluate(lm, tasks=["humaneval"], confirm_run_unsafe_code=True)
# print("humaneval done")
# _ = lm_eval.simple_evaluate(lm, tasks=["xnli"])
# print("xnli done")
# _ = lm_eval.simple_evaluate(lm, tasks=["arc_challenge_mt"])
# print("arc_mt done")
_ = lm_eval.simple_evaluate(lm, tasks=["mmlu_pro"])
print("mmlupro done")

# _ = lm_eval.simple_evaluate(lm, tasks=["mmlu"])
# _ = lm_eval.simple_evaluate(lm, tasks=["niah_single_1"])
# _ = lm_eval.simple_evaluate(lm, tasks=["niah_single_2"])
# _ = lm_eval.simple_evaluate(lm, tasks=["niah_single_3"])
# _ = lm_eval.simple_evaluate(lm, tasks=["ruler_vt"])
# _ = lm_eval.simple_evaluate(lm, tasks=["ruler_qa_squad"])
# _ = lm_eval.simple_evaluate(lm, tasks=["swde"])
# _ = lm_eval.simple_evaluate(lm, tasks=["fda"])
# _ = lm_eval.simple_evaluate(lm, tasks=["hellaswag"])
"""
_ = lm_eval.simple_evaluate(lm, tasks=["lambada"])
_ = lm_eval.simple_evaluate(lm, tasks=["openbookqa"])
_ = lm_eval.simple_evaluate(lm, tasks=["squadv2"])
_ = lm_eval.simple_evaluate(lm, tasks=["squad_completion"])
_ = lm_eval.simple_evaluate(lm, tasks=["nq_open"])
_ = lm_eval.simple_evaluate(lm, tasks=["drop"])
_ = lm_eval.simple_evaluate(lm, tasks=["mmlu"])
_ = lm_eval.simple_evaluate(lm, tasks=["triviaqa"])
_ = lm_eval.simple_evaluate(lm, tasks=["arc_easy"])
_ = lm_eval.simple_evaluate(lm, tasks=["arc_challenge"])
_ = lm_eval.simple_evaluate(lm, tasks=["piqa"])
_ = lm_eval.simple_evaluate(lm, tasks=["squadv2"])
_ = lm_eval.simple_evaluate(lm, tasks=["winogrande"])
"""
