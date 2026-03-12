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

**Tarefa 1 — Máscara Causal (Look-Ahead Mask)**  
Função `create_causal_mask(seq_len)` que impede o modelo de olhar para tokens futuros durante o treinamento.
