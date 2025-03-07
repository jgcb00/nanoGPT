from typing import Union, List, Optional
import tqdm
import tiktoken

import torch
import torch.nn.functional as F

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

from arch.gpt import GPT
from arch.dragon import Dragon

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

BSZ_FOR_TASKS = {
    "hellaswag": 128,
    "swde": 32,
    "squadv2": 32,
    "fda": 32,
    "nq_open": 32,
    "mmlu": 32,
    "triviaqa": 64,
    "arc_easy": 64,
    "arc_challenge": 64,
    "piqa": 64,
    "winogrande": 64,
}

class NanoLM(LM):
    def __init__(self, model: Union[GPT, Dragon] = None, enc: tiktoken.core.Encoding = None, batch_size: int = 32):
        super().__init__()

        assert model.config.rmsnorm == False, "rmsnorm must be False for Mamba2 inference for now."

        self.model = model
        self.config = model.config
        self.enc = enc

        self.batch_size = batch_size
    
    @torch.no_grad()
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        example : 
        input: ('Roof shingle removal: A man is sitting on a roof. He', ' is using wrap to wrap a pair of skis.')
        returns: (loglikelihood of target, is_greedy ie whether decoding greedily gives the target)
        """

        task = requests[0].task_name
        if task in BSZ_FOR_TASKS:
            self.batch_size = BSZ_FOR_TASKS[task]

        outputs = []

        # loglikelihood computation
        lls = []
        if task in ["hellaswag"]: # skip the likelihood for the tasks we know we don't need it for
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
        for i in tqdm.tqdm(range(0, len(requests), self.batch_size)):
            batch = requests[i:i+self.batch_size]

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
                generated_batch = self.generate(prompts=prompts, n_tokens=n_tokens, samples=False) # list of B x (L) tensors
            
            for i in range(len(batch)):
                generated = generated_batch[i].tolist()
                is_greedy = int(generated == target_enc_list[i])
                is_greedys.append(is_greedy)

        for i in range(len(requests)):
            outputs.append((lls[i], is_greedys[i]))
        return outputs

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        """
        example :
        input: ('this is the beginning of the', {'until': ['.']})
        returns: 'end'
        """

        task = requests[0].task_name
        if task in BSZ_FOR_TASKS:
            self.batch_size = BSZ_FOR_TASKS[task]

        outputs = []
        for i in tqdm.tqdm(range(0, len(requests), self.batch_size)):
            batch = requests[i:i+self.batch_size]

            prompts = []
            n_tokens_list = []
            samples = []
            temperatures = []
            top_ks = []
            stop_tokens_list = []

            for request in batch:
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
                if 'top_k' in kwargs:
                    top_k = kwargs['top_k']
                else:
                    top_k = None

                input_enc = self.enc.encode(input_str)

                prompts.append(torch.tensor(input_enc))
                n_tokens_list.append(max_gen_toks)
                samples.append(do_sample)
                temperatures.append(temperature)
                top_ks.append(top_k)
                stop_tokens_list.append(stop_tokens)

            with ctx:
                generated_batch = self.generate(prompts=prompts, n_tokens=n_tokens_list, samples=samples, temperatures=temperatures, top_ks=top_ks, stop_tokens=stop_tokens_list) # list of B (L) tensors
            
            for i in range(len(batch)):
                generated = generated_batch[i]
                generated_str = self.enc.decode(generated.tolist())
                outputs.append(generated_str)

        return outputs

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        print("loglikelihood_rolling not implemented.")
        return
    
    """ naive function, doesnt use any cache and shouldnt use. all prompts should have the same length. it is kept for reference. """
    @torch.no_grad()
    def generate_naive(self, prompt, n_tokens: int, sample: bool = True, top_k: int = None, temperature: float = 1.0):
        # prompt: (b, T) tensor
        # outputs: (b, t) tensor

        if top_k is not None:
            top_k = min(top_k, self.config.vocab_size)
        
        input_device = prompt.device
        prompt = prompt.to(self.model.transformer.wte.weight.device)

        self.model.eval()
        generated = prompt.clone()

        for _ in range(n_tokens):
            logits = self.model.forward(generated, targets=None, caches=None) # (B, L, vocab_size)
            next_token_logits = logits[:, -1]

            if sample:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                    
                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k) # (B, k) ordered from lowest to biggest
                    probs[probs < values[:, -1, None]] = 0 # zero-out all probs except the k first
                    probs = probs / probs.sum(axis=1, keepdims=True)

                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
            generated = torch.cat([generated, next_token], dim=1)

        self.model.train()

        return generated.to(input_device)[:, -n_tokens:]
    
    """ optimized function, that uses cache and works with batched prompts (of different lengths) """
    @torch.no_grad()
    def generate(self, prompts, n_tokens: List[int], samples: Union[bool, List[bool]] = True, top_ks: Union[None, int, List[Optional[int]]] = None, temperatures: Union[float, List[float]] = 1.0, stop_tokens: List[Optional[List[int]]] = None):
        # prompts : list of B x (L) tensors

        B = len(prompts)
        min_len = min(prompt.size(0) for prompt in prompts)
        max_len = max(prompt.size(0) for prompt in prompts)
        max_num_tokens = max(n_tokens)
        max_len_generation = max([p.size(0) + nt for (p, nt) in zip(prompts, n_tokens)]) # max timestep that wil be reached during generation
        
        if not isinstance(samples, list):
            samples = [samples] * B
        else:
            assert len(samples) == B, "Length of samples must equal number of prompts"
        if not isinstance(top_ks, list):
            top_ks = [top_ks] * B
        else:
            assert len(top_ks) == B, "Length of top_ks must equal number of prompts"
        if not isinstance(temperatures, list):
            temperatures = [temperatures] * B
        else:
            assert len(temperatures) == B, "Length of temperatures must equal number of prompts"
        for tk in top_ks:
            if tk is not None:
                assert tk <= self.config.vocab_size_real, "top_k must be <= vocab_size"

        input_device = prompts[0].device
        model_device = self.model.transformer.wte.weight.device
        
        self.model.eval()
        
        padded_prompts = [F.pad(prompt, (0, max_len-prompt.size(0))) for prompt in prompts]
        padded_prompts = torch.stack(padded_prompts)
        
        batched_generated = torch.zeros(B, max_len+max_num_tokens, dtype=torch.long, device=model_device)
        batched_generated[:, :max_len] = padded_prompts
        
        prompt_lengths = torch.tensor([p.size(0) for p in prompts], device=input_device)
        position_ids = torch.arange(max_len+max_num_tokens, device=input_device).unsqueeze(0).expand(B, -1)
        active_mask = (position_ids < prompt_lengths.unsqueeze(1)).to(model_device)

        if stop_tokens is None:
            stop_tokens = [[] for _ in range(B)]
        else:
            if len(stop_tokens) != B:
                raise ValueError("stop_tokens must have the same length as prompts")
            stop_tokens = [st if st is not None else [] for st in stop_tokens]
        finished = [False] * B

        # caches is a list of cache, one per layer
        # cache is composed of : - if Mamba(2) layer : the hidden state, and the last d_conv-1 inputs (see more in mamba_lm.py)
        #                        - if attention layer : the KV cache, ie 2 tensors of shape (B, num_kv_heads, L, head_dim)
        caches = [block.get_empty_cache() for block in self.model.transformer.h]

        # process prompt in one go
        logits, caches = self.model.forward(batched_generated[:, :min_len], targets=None, caches=caches) # (B, L, vocab_size)
        next_token_logits = logits[:, -1] # (B, vocab_size)
        next_token_logits[:, self.config.vocab_size_real:] = -float("inf") # mask out the tokens that are not in the real vocab (used for efficiency reasons)

        # generate one token at a time
        for t in range(min_len, max_len_generation):
            new_tokens_list = []

            for i in range(B):
                if samples[i]:
                    prob = F.softmax(next_token_logits[i] / temperatures[i], dim=-1)
                    if top_ks[i] is not None:
                        values, _ = torch.topk(prob, k=top_ks[i])
                        threshold = values[-1]
                        prob = torch.where(prob < threshold, torch.tensor(0.0, device=prob.device), prob)
                        prob = prob / (prob.sum() + 1e-8)
                    token_i = torch.multinomial(prob, num_samples=1)
                else:
                    token_i = next_token_logits[i].argmax().unsqueeze(0)
                new_tokens_list.append(token_i)
            next_token = torch.stack(new_tokens_list, dim=0) # (B, 1)
            tokens_generated = next_token.squeeze(1) # (B,)

            # For finished sequences, reuse the previous token.
            finished_mask = torch.tensor(finished, device=model_device)
            prev_token = batched_generated[:, t-1]
            new_tokens = torch.where(finished_mask, prev_token, tokens_generated)

            update_mask = (~active_mask[:, t]) & torch.tensor([not f for f in finished], device=model_device)
            batched_generated[:, t] = torch.where(update_mask, new_tokens, batched_generated[:, t])

            # Mark finished for sequences that just produced a stop token.
            for i in range(B):
                if not finished[i] and stop_tokens[i] and (tokens_generated[i].item() in stop_tokens[i]):
                    finished[i] = True

            # Break early only if every sequence has stop tokens and is finished.
            if all(len(st) > 0 for st in stop_tokens) and all(finished):
                break

            next_token_logits, caches = self.model.forward(batched_generated[:, [t]], targets=None, caches=caches) # (B, 1, vocab_size), caches
            next_token_logits = next_token_logits.squeeze(1) # (B, vocab_size)
            next_token_logits[:, self.config.vocab_size_real:] = -float("inf") # mask out the tokens that are not in the real vocab (used for efficiency reasons)

        self.model.train()

        generated = []
        for i, (seq, nt) in enumerate(zip(batched_generated, n_tokens)):
            start = prompts[i].size(0)
            gen_seq = seq[start: start + nt].to(input_device)
            if stop_tokens[i]:
                tokens = gen_seq.tolist()
                cut_idx = next((idx for idx, token in enumerate(tokens) if token in stop_tokens[i]), None)
                if cut_idx is not None:
                    gen_seq = gen_seq[:cut_idx]
            generated.append(gen_seq)
        return generated
