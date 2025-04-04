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
    def __init__(self, model: GPT = None, enc: tiktoken.core.Encoding = None, batch_size: int = 32):
        super().__init__()

        self.model = model
        self.model.eval()
        self.enc = enc

        self.eval_task = None
        self.batch_size = batch_size
    
    @torch.no_grad()
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        example : 
        input: ('Roof shingle removal: A man is sitting on a roof. He', ' is using wrap to wrap a pair of skis.')
        returns: (loglikelihood of target, is_greedy ie whether decoding greedily gives the target)
        """
        
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
        """
        example :
        input: ('this is the beginning of the', {'until': ['.']})
        returns: 'end'
        """

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
                generated_batch = self.model.generate(prompts=prompts, n_tokens=n_tokens_list, samples=samples, temperatures=temperatures, top_ks=top_ks, stop_tokens=stop_tokens_list) # list of B (L) tensors
            
            for i in range(len(batch)):
                generated = generated_batch[i]
                generated_str = self.enc.decode(generated.tolist())
                outputs.append(generated_str)

        return outputs

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        print("loglikelihood_rolling not implemented.")
        return

config = NanoConfig()
config.d_model = 768
config.n_heads = 12
config.n_layers = 12
config.vocab_size = 50304
config.vocab_size_real = 50257

model = GPT(config)
model.cuda()
#model = torch.compile(model, dynamic=False)

with open('enc.pkl', 'rb') as f:
    enc_pickled = pickle.load(f)
enc = tiktoken.core.Encoding(enc_pickled.pop('name'), **enc_pickled)

lm = GPT_LM(model, enc, batch_size=32)

# tasks we care about: hellaswag, swde, squadv2, squad_completion, fda, nq_open, drop
# + : mmlu, triviaqa, arc, piqa, winogrande

#results = lm_eval.simple_evaluate(lm, tasks=["hellaswag"]) # loglikelihood
#results = lm_eval.simple_evaluate(lm, tasks=["swde"]) # generate_until

task = "squadv2"
results = lm_eval.simple_evaluate(lm, tasks=[task])

# save results to json
import json
with open(f'results_{task}.json', 'w') as f:
    json.dump(results['results'], f)

print(results['results'])
