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
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k    = K.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    return softmax(scores) @ V

def create_causal_mask(seq_len):
    return np.where(np.tril(np.ones((seq_len, seq_len))), 0.0, -np.inf)


def cross_attention(encoder_out, decoder_state):
    WQ = np.random.randn(D_MODEL, D_MODEL) * np.sqrt(2.0 / D_MODEL)
    WK = np.random.randn(D_MODEL, D_MODEL) * np.sqrt(2.0 / D_MODEL)
    WV = np.random.randn(D_MODEL, D_MODEL) * np.sqrt(2.0 / D_MODEL)
    Q  = decoder_state @ WQ
    K  = encoder_out   @ WK
    V  = encoder_out   @ WV
    return scaled_dot_product_attention(Q, K, V, mask=None)

W_out = np.random.randn(D_MODEL, VOCAB_SIZE) * 0.01

def generate_next_token(current_sequence_ids, encoder_out):
    seq_len         = len(current_sequence_ids)
    embedding_table = np.random.randn(VOCAB_SIZE, D_MODEL) * 0.1
    decoder_input   = embedding_table[current_sequence_ids][np.newaxis, :, :]

    WQ_sa = np.random.randn(D_MODEL, D_MODEL) * np.sqrt(2.0 / D_MODEL)
    WK_sa = np.random.randn(D_MODEL, D_MODEL) * np.sqrt(2.0 / D_MODEL)
    WV_sa = np.random.randn(D_MODEL, D_MODEL) * np.sqrt(2.0 / D_MODEL)

    mask     = create_causal_mask(seq_len)
    self_att = scaled_dot_product_attention(
        decoder_input @ WQ_sa,
        decoder_input @ WK_sa,
        decoder_input @ WV_sa,
        mask=mask
    )
    dec_state   = self_att + decoder_input
    cross_out   = cross_attention(encoder_out, dec_state)
    last_hidden = cross_out[0, -1, :]
    return softmax(last_hidden @ W_out)

print("=" * 55)
print("TAREFA 1 — Máscara Causal (Look-Ahead Mask)")
print("=" * 55)

seq_len = 5
M       = create_causal_mask(seq_len)
print("\nMáscara M:")
print(M)

X_dummy       = np.random.randn(1, seq_len, D_MODEL)
Q_dummy       = X_dummy @ np.random.randn(D_MODEL, 64)
K_dummy       = X_dummy @ np.random.randn(D_MODEL, 64)
scores_masked = Q_dummy @ K_dummy.transpose(0, 2, 1) / np.sqrt(64) + M
attn_weights  = softmax(scores_masked)

print("\nPesos de atenção após Softmax com máscara:")
print(np.round(attn_weights[0], 4))
print("\n→ Posições futuras são estritamente 0.0 ✓")


print("\n" + "=" * 55)
print("TAREFA 2 — Cross-Attention (Ponte Encoder-Decoder)")
print("=" * 55)

encoder_output = np.random.randn(1, 10, D_MODEL)
decoder_state  = np.random.randn(1,  4, D_MODEL)

print(f"\nencoder_output : {encoder_output.shape}")
print(f"decoder_state  : {decoder_state.shape}")

cross_out = cross_attention(encoder_output, decoder_state)
print(f"Saída Cross-Attention : {cross_out.shape} ✓")


print("\n" + "=" * 55)
print("TAREFA 3 — Loop de Inferência Auto-Regressivo")
print("=" * 55)

MAX_STEPS   = 20
EOS_STEP    = 5
current_ids = [TOKEN_START]

print(f"\nSequência inicial : {[id_to_token[i] for i in current_ids]}\n")

step = 0
while True:
    step     += 1
    probs     = generate_next_token(current_ids, encoder_output)
    next_id   = TOKEN_EOS if step == EOS_STEP else int(np.argmax(probs))
    next_token = id_to_token[next_id]
    current_ids.append(next_id)

    print(f"  Passo {step:02d} → '{next_token}'  (prob={probs[next_id]:.4f})")
