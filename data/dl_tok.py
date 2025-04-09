import tiktoken_ext.openai_public
import pickle

enc = tiktoken_ext.openai_public.gpt2()

with open('data/enc.pkl', 'wb') as file:
    pickle.dump(enc, file)
