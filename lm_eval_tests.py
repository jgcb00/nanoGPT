import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

import tqdm
import tiktoken
import pickle

import torch

from config import NanoConfig
from arch.gpt import GPT

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

class GPT_LM(LM):
    def __init__(self, model: GPT = None, enc: tiktoken.core.Encoding = None):
        super().__init__()

        self.model = model
        self.model.eval()
        self.enc = enc

        self.eval_task = None
    
    @torch.no_grad()
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        example : 
        input: ('Roof shingle removal: A man is sitting on a roof. He', ' is using wrap to wrap a pair of skis.')
        returns: (loglikelihood of target, is_greedy, ie whether decoding greedily gives the target)
        """

        batch_size = 32
        
        outputs = []

        # loglikelihood computation
        lls = []
        if self.eval_task in ["hellaswag"]: # skip the likelihood for the tasks we don't need it for
            for request in requests:
                lls.append(0.)
        else:    
            for request in tqdm.tqdm(requests):
                input_str, target_str = request.args

                input_enc = self.enc.encode(input_str) # list of ints
                target_enc = self.enc.encode(target_str)
                len_input = len(input_enc)
                len_target = len(target_enc)

                prompt = torch.tensor(input_enc+target_enc, dtype=torch.long, device=self.model.transformer.wte.weight.device).unsqueeze(0)
                x = prompt[:, :-1]
                y = prompt[:, 1:].clone()
                y[:, :len_input-1] = -1

                with ctx:
                    loss = self.model(x, targets=y)
                loglikelihood = -loss.item()
                del loss

                lls.append(loglikelihood)
        
        # is_greedy computation
        is_greedys = []
        for i in tqdm.tqdm(range(0, len(requests), batch_size)):
            batch = requests[i:i+batch_size]

            prompts = []
            n_tokens = []
            target_enc_list = []
            for request in batch:
                input_str, target_str = request.args

                input_enc = self.enc.encode(input_str) # list of ints
                target_enc = self.enc.encode(target_str)
                len_target = len(target_enc)

                prompts.append(torch.tensor(input_enc))
                n_tokens.append(len_target)
                target_enc_list.append(target_enc)

            with ctx:
                generated_batch = self.model.generate(prompts=prompts, n_tokens=n_tokens, sample=False) # list of B x (L) tensors
            
            for i in range(len(batch)):
                generated = generated_batch[i].tolist()
                is_greedy = int(generated == target_enc_list[i])
                is_greedys.append(is_greedy)

        for i in range(len(requests)):
            outputs.append((lls[i], is_greedys[i]))
        return outputs

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        print("generate_until")
        print(len(requests))
        print(requests[0])
        print(requests[0].args) # tuple (input, {'until': ['.', '\n\n]', 'max_gen_toks': 128, 'do_sample': False, 'temperature': 0.})

        compteur = 0
        no_maxgen = 0
        max_toks = []
        for request in tqdm.tqdm(requests):
            input_str, kwargs = request.args

            if 'max_gen_toks' in kwargs:
                max_toks.append(kwargs['max_gen_toks'])
            else:
                no_maxgen += 1

            compteur += 1

        print(compteur)
        print(no_maxgen)
        print(f"mean max_toks: {sum(max_toks)/len(max_toks)}")
        print(f"max max_toks: {max(max_toks)}")
        print(f"min max_toks: {min(max_toks)}")

        # for each request, return a string, ie the completion of the model

        outputs = []
        for request in tqdm.tqdm(requests):
            input_str, kwargs = request.args

            if 'until' in kwargs:
                until = kwargs['until']
                stop_tokens = [self.enc.encode(token)[0] for token in until]
            else:
                stop_tokens = None
            
            if 'max_gen_toks' in kwargs:
                max_gen_toks = kwargs['max_gen_toks']
            else:
                max_gen_toks = 1024
            
            if 'do_sample' in kwargs:
                do_sample = kwargs['do_sample']
            else:
                do_sample = True

            if 'temperature' in kwargs:
                temperature = kwargs['temperature']
            else:
                temperature = 1.

            input_enc = self.enc.encode(input_str)

            with ctx:
                generated = self.model.generate(prompts=[torch.tensor(input_enc)], n_tokens=[max_gen_toks], sample=do_sample, temperature=temperature, stop_tokens=[stop_tokens]) # list of B x (L) tensors

            print(generated)

            break

            outputs.append()
        return outputs

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        print("loglikelihood_rolling not implemented.")
        return

#setup_logging("DEBUG")

config = NanoConfig()
config.d_model = 768
config.n_heads = 12
config.n_layers = 12
config.vocab_size = 50304

model = GPT(config)
model.cuda()
#model = torch.compile(model, dynamic=False)

with open('enc.pkl', 'rb') as f:
    enc_pickled = pickle.load(f)
enc = tiktoken.core.Encoding(enc_pickled.pop('name'), **enc_pickled)

lm = GPT_LM(model, enc)

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