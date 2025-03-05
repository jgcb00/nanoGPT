import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.utils import setup_logging

from config import NanoConfig
from arch.gpt import GPT

class GPT_LM(LM):
    def __init__(self, model: GPT):
        super().__init__()

        self.model = model
        
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        Each request contains Instance.args : Tuple[str, str] containing 1. an input string to the LM and 2. a target string on which the loglikelihood of the LM producing this target, conditioned on the input, will be returned.
        Each request will have, as result, (ll, is_greedy): Tuple[float, int] returned, where ll is a floating point number representing the log probability of generating the target string conditioned on the input, and is_greedy being either the value 0 or 1, with it being 1 if and only if the target string would be generated by greedy sampling from the LM (that is, if the target string is the most likely N-token string to be output by the LM given the input. )
        """
        print("loglikelihood ===================")
        print(len(requests)) # 40168 for hellaswag
        print(requests[0])
        print(requests[0].args) # ('Roof shingle removal: A man is sitting on a roof. He', ' is using wrap to wrap a pair of skis.')
        print("=================================")

        output = []
        for request in requests:
            output.append((-10.0, 1)) # for each request, return a tuple (ll, is_greedy)
        return output

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        print("loglikelihood_rolling")
        return
    
    def generate_until(self, requests: list[Instance]) -> list[str]:
        print("generate_until")
        print(len(requests))
        print(requests[0])
        print(requests[0].args) # tuple (input, {'until': ['.', '\n\n]', 'max_gen_toks': 128, 'do_sample': False, 'temperature': 0.})

        # for each request, return a string
        return

#setup_logging("DEBUG")

config = NanoConfig()
config.d_model = 128
config.n_heads = 4
config.n_layers = 3
config.vocab_size = 50304

model = GPT(config)
lm = GPT_LM(model)

#results = lm_eval.simple_evaluate(lm, tasks=["hellaswag"])
results = lm_eval.simple_evaluate(lm, tasks=["swde"])

print(results['results'])

"""
import json
with open("res.json", "w") as file:
    json.dump(results, file)
"""

"""
@positional_deprecated
def simple_evaluate(
    model,
    model_args: Optional[Union[str, dict]] = None,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[Union[int, str]] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[str] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    evaluation_tracker: Optional[EvaluationTracker] = None,
    system_instruction: Optional[str] = None,
    apply_chat_template: Union[bool, str] = False,
    fewshot_as_multiturn: bool = False,
    gen_kwargs: Optional[str] = None,
    task_manager: Optional[TaskManager] = None,
    verbostiy=None,
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
    confirm_run_unsafe_code: bool = False,
):
    Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str, dict]
        String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param cache_requests: bool, optional
        Speed up evaluation by caching the building of dataset requests. `None` if not caching.
    :param rewrite_requests_cache: bool, optional
        Rewrites all of the request cache if set to `True`. `None` if not desired.
    :param delete_requests_cache: bool, optional
        Deletes all of the request cache if set to `True`. `None` if norunpt desired.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderrs. set to 0 for no stderr calculations to be performed.
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: Union[bool, str]
        Specifies whether to apply a chat template to the prompt.
        - If set to True, the default chat template is applied.
        - If set to a string, applies the specified chat template by name.
        Defaults to False (no chat template applied).
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param verbostiy: str
        Verbosity level for logging
    :param predict_only: bool
        If true only model outputs will be generated and returned. Metrics will not be evaluated
    :param random_seed: int
        Random seed for python's random module. If set to None, the seed will not be set.
    :param numpy_random_seed: int
        Random seed for numpy. If set to None, the seed will not be set.
    :param torch_random_seed: int
        Random seed for torch. If set to None, the seed will not be set.
    :param fewshot_random_seed: int
        Random seed for fewshot sampler random generator. If set to None, the seed of generator will be set to None.

    :return
        Dictionary of results
"""