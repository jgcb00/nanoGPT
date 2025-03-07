import tiktoken
import pickle

with open('enc.pkl', 'rb') as f:
    enc_pickled = pickle.load(f)

enc = tiktoken.core.Encoding(enc_pickled.pop('name'), **enc_pickled)

encoded = enc.encode("Hello, world!") # [15496, 11, 995]
print(encoded)