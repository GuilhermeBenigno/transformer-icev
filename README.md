Descrição

Implementação dos blocos matemáticos centrais do Decoder Transformer usando apenas `numpy`, conforme o paper *"Attention Is All You Need"* (Vaswani et al., 2017).

---

Como Rodar

Pré-requisitos

```bash
pip install numpy
```

 Execução

```bash
python transformer_decoder.py
```

---

O que é implementado

Tarefa 1 — Máscara Causal (Look-Ahead Mask)**  
Função `create_causal_mask(seq_len)` que impede o modelo de olhar para tokens futuros durante o treinamento.

Tarefa 2 — Cross-Attention (Ponte Encoder-Decoder)**  
Função `cross_attention(encoder_out, decoder_state)` onde Q vem do Decoder e K/V vêm do Encoder.

Tarefa 3 — Loop de Inferência Auto-Regressivo**  
Função `generate_next_token()` chamada iterativamente em um loop `while` que para ao gerar o token `<EOS>`.

---

Nota de Crédito

Ferramenta de IA (Claude – Anthropic) foi consultada como apoio para revisão de sintaxe NumPy. O código foi entendido, adaptado e validado pelo aluno.
