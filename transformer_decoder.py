import numpy as np

np.random.seed(42)

D_MODEL   = 512
VOCAB_SIZE = 10_000

TOKEN_START = 0
TOKEN_EOS   = 1

id_to_token = {TOKEN_START: "<START>", TOKEN_EOS: "<EOS>"}
for i in range(2, VOCAB_SIZE):
    id_to_token[i] = f"palavra_{i}"

def softmax(x):
    """Softmax numericamente estável no último eixo."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q, K, V) = softmax(QKᵀ/√d_k + M) V
