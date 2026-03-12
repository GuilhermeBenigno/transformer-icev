import numpy as np

np.random.seed(42)

D_MODEL   = 512
VOCAB_SIZE = 10_000

TOKEN_START = 0
TOKEN_EOS   = 1

id_to_token = {TOKEN_START: "<START>", TOKEN_EOS: "<EOS>"}
for i in range(2, VOCAB_SIZE):
    id_to_token[i] = f"palavra_{i}"
