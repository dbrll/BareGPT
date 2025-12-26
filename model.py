import numpy as np

import engine
import utils

"""
BareGPT: A minimalist, NumPy-only implementation of a GPT-style Transformer.

Inspired by NanoGPT (Karpathy, 2023) and GPT-1 (Radford et al., 2018) for the
decoder-only generative approach, and Attention Is All You Need
(Vaswani et al., 2017) for the core Transformer mechanics

Damien Boureille, 2025
MIT Licence
"""


def init_weights(d_model, block_size, vocab_size, n_heads, n_layers):
    """
    Initializes all model weights for a multi-layer Transformer.
    """
    params = {
        "n_heads": n_heads,
        "d_model": d_model,
        "block_size": block_size,
        "vocab_size": vocab_size,
        "n_layers": n_layers,
    }

    # Initialize weights for each Transformer block
    for i in range(n_layers):
        # 1. Attention weights for layer i
        params[f"W_q_{i}"] = utils.xavier_init((d_model, d_model))
        params[f"W_k_{i}"] = utils.xavier_init((d_model, d_model))
        params[f"W_v_{i}"] = utils.xavier_init((d_model, d_model))
        params[f"W_o_{i}"] = utils.xavier_init((d_model, d_model))

        # 2. MLP weights for layer i (d_model -> 4 * d_model -> d_model)
        hidden_dim = 4 * d_model  # Hidden layer should be 4 times the size of d_model
        W1 = utils.xavier_init((d_model, hidden_dim))
        b1 = np.zeros((1, hidden_dim))
        W2 = utils.xavier_init((hidden_dim, d_model))
        b2 = np.zeros((1, d_model))

        params[f"W1_{i}"] = W1
        params[f"b1_{i}"] = b1
        params[f"W2_{i}"] = W2
        params[f"b2_{i}"] = b2

        # 3. LayerNorm parameters per layer
        params[f"ln1_gamma_{i}"] = np.ones((1, d_model))
        params[f"ln1_beta_{i}"] = np.zeros((1, d_model))
        params[f"ln2_gamma_{i}"] = np.ones((1, d_model))
        params[f"ln2_beta_{i}"] = np.zeros((1, d_model))

    # Output head weights (global)
    params["W_out"] = utils.xavier_init((d_model, vocab_size))
    params["b_out"] = np.zeros((vocab_size,))

    # Token and position embedding weights (global)
    params["token_embed"] = np.random.randn(vocab_size, d_model) * 0.01
    params["pos_embed"] = np.random.randn(block_size, d_model) * 0.01

    return params


def embed_tokens_and_positions(X_ids, params):
    """
    Computes the sum of token and positional embeddings.
    Args:
        X_ids: Input token indices of shape (batch_size, seq_len)
        params: Dictionary containing 'token_embed' and 'pos_embed'

    Returns:
        X: Combined embeddings of shape (batch_size, seq_len, d_model)
    """
    # Map token IDs to their corresponding embedding vectors
    # Shape transition: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
    token_embed = params["token_embed"]
    tok_emb = token_embed[X_ids]  # (block_size, d_model)

    # Generate a sequence of position indices [0, 1, ..., seq_len-1]
    pos_embed = params["pos_embed"]
    batch, seq_len = X_ids.shape
    positions = np.arange(seq_len)

    # Retrieve positional vectors for the current sequence length
    # Shape: (seq_len, d_model)
    pos_emb_block = pos_embed[positions]  # (block_size, d_model)

    # Expand dimensions for broadcasting across the batch
    # Shape: (1, seq_len, d_model)
    pos_emb_block = pos_emb_block[None, :, :]

    # Element-wise sum of token and position information
    # The model learns to integrate "what" (token) with "where" (position)
    X = tok_emb + pos_emb_block
    return X


def layernorm_forward(x, eps=1e-5):
    """
    Layer Normalization: Stabilizes the network by normalizing inputs across features.
    Formula: y = (x - E[x]) / sqrt(Var[x] + eps)
    """

    # Moyenne et variance sur les features
    mean = np.mean(x, axis=-1, keepdims=True)  # (B, T, 1)
    var = np.var(x, axis=-1, keepdims=True)  # (B, T, 1)

    inv_std = 1.0 / np.sqrt(var + eps)
    x_norm = (x - mean) * inv_std

    cache = {
        "x": x,
        "mean": mean,
        "var": var,
        "inv_std": inv_std,
        "eps": eps,
    }
    return x_norm, cache


def multi_head_attention(X, params, layer_idx):
    """
    Computes Multi-Head Masked Self-Attention for a specific layer.

    Args:
        X: Input tensor of shape (B, T, D)
        params: Dictionary containing all model weights
        layer_idx: The index of the current transformer layer

    Tensor shape:
        B (Batch Size): number of independent sequences processed simultaneously
        T (Time / Sequence Length): number of tokens per sequence (temporal window)
        D (Dimension / d_model): size of the embedding vector for each token
        H (Head): number of attention heads
    """
    B, T, D = X.shape
    H = params["n_heads"]

    # 1. Retrieve weights for this specific layer
    W_q = params[f"W_q_{layer_idx}"]
    W_k = params[f"W_k_{layer_idx}"]
    W_v = params[f"W_v_{layer_idx}"]
    W_o = params[f"W_o_{layer_idx}"]

    Hd = D // H

    # 2. Linear projections -> (B, T, D)
    # Q (Query): What the token is looking for
    # K (Key): What the token contains (its index for others to find it)
    # V (Value): The actual information the token communicates if it matches a Query
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # 3. Reshape and transpose for multi-head: (B, H, T, Hd)
    Qh = Q.reshape(B, T, H, Hd).transpose(0, 2, 1, 3)
    Kh = K.reshape(B, T, H, Hd).transpose(0, 2, 1, 3)
    Vh = V.reshape(B, T, H, Hd).transpose(0, 2, 1, 3)

    # 4. Scaled dot-product scores: (B, H, T, T)
    # (B, H, T, Hd) @ (B, H, Hd, T) -> (B, H, T, T)
    #
    # This is the most famous formula in "Attention is All You Need"
    # Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V
    # This computes how much focus each token should put on every other token
    #
    scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / np.sqrt(Hd)

    # 5. Causal masking
    # Ensures the model can't "cheat" by looking at future tokens
    # Sets future scores to -infinity so they become 0 after softmax
    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores_masked = np.where(causal_mask[None, None, :, :], -1e10, scores)

    # 6. Softmax
    # Normalizes raw scores into probabilities that sum to 1
    # This determines the focus or weight given to each past token
    scores_max = np.max(scores_masked, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_masked - scores_max)
    att_probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 7. Weighted sum of values: (B, H, T, Hd)
    # Replaced np.einsum with np.matmul: (B, H, T, T) @ (B, H, T, Hd) -> (B, H, T, Hd)
    att_out_h = att_probs @ Vh

    # 8. Concatenate heads and final projection -> (B, T, D)
    att_out = att_out_h.transpose(0, 2, 1, 3).reshape(B, T, D)
    att_out = att_out @ W_o

    # 9. Cache now stores the specific weights used
    cache = {
        "X": X,
        "Q": Q,
        "K": K,
        "V": V,
        "Qh": Qh,
        "Kh": Kh,
        "Vh": Vh,
        "att_probs": att_probs,
        "scores": scores,
        "scores_masked": scores_masked,
        "causal_mask": causal_mask,
        "W_q": W_q,
        "W_k": W_k,
        "W_v": W_v,
        "W_o": W_o,
        "n_heads": H,
        "head_dim": Hd,
        "layer_idx": layer_idx,
    }

    return att_probs, att_out, cache


def mlp_forward(X, params, layer_idx):
    """
    Position-wise Feed-Forward Network for a specific layer.
    Structure: Linear (Z1) -> GELU (A1) -> Linear (Z2)
    """

    # 1. Retrieve indexed weights for this specific layer
    W1 = params[f"W1_{layer_idx}"]
    b1 = params[f"b1_{layer_idx}"]
    W2 = params[f"W2_{layer_idx}"]
    b2 = params[f"b2_{layer_idx}"]

    # 2. Linear 1: Expansion from d_model to 4 * d_model
    Z1 = X @ W1 + b1

    # 3. GELU activation approximation
    # GELU is smoother than ReLU and helps gradients flow better in deep networks
    A1 = 0.5 * Z1 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (Z1 + 0.044715 * Z1**3)))

    # 4. Linear 2: Projection back to d_model
    Z2 = A1 @ W2 + b2
    out = Z2

    # 5. Cache for the backward pass
    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "layer_idx": layer_idx,
    }

    return out, cache


def transformer_block(X, params, layer_idx):
    """
    Executes one full Transformer layer (Block i) including Self-Attention and MLP.

    Flow:
        1. LayerNorm -> Multi-Head Attention -> Residual Connection
        2. LayerNorm -> Feed-Forward (MLP)   -> Residual Connection

    Args:
        X (ndarray): Input hidden states of shape (Batch, Time, Dimension).
        params (dict): Global dictionary containing all model weights.
                       Uses 'layer_idx' to fetch layer-specific parameters.
        layer_idx (int): The index of this layer in the stack (0 to n_layers-1).

    Returns:
        Y (ndarray): Output hidden states of shape (Batch, Time, Dimension).
        cache_block (dict): Intermediate tensors (X, X2, norms, etc.) stored for
                            the backward pass to compute gradients efficiently.
        att_matrix (ndarray): The attention weight matrix, useful for visualization.
    """

    # 1. Self-Attention Sub-layer
    X_norm, ln1_cache = layernorm_forward(X)

    # We pass layer_idx so attention knows which W_q_i, W_k_i... to take
    att_probs, att_matrix, att_cache = multi_head_attention(X_norm, params, layer_idx)
    X2 = X + att_matrix  # Residual

    # 2. Feed-Forward Sub-layer
    X2_norm, ln2_cache = layernorm_forward(X2)

    # We pass layer_idx so mlp knows which W1_i, W2_i... to take
    mlp_out, mlp_cache = mlp_forward(X2_norm, params, layer_idx)
    Y = X2 + mlp_out  # Y = Second residual

    # Cache for the backward pass
    cache_block = {
        "X": X,
        "ln1": ln1_cache,
        "att": att_cache,
        "X2": X2,
        "ln2": ln2_cache,
        "mlp": mlp_cache,
        "layer_idx": layer_idx,  # For backward to know which layer it is
    }

    return Y, cache_block, att_matrix


def forward(X, targets, params, X_ids):
    """
    Performs a full forward pass through the multi-layer Transformer model.

    Flow:
        Input Embeddings -> [Transformer Block * N] -> Linear Output Head -> Softmax -> Loss

    Args:
        X (ndarray):
            Input embeddings of shape (Batch, Time, Dimension).
            Usually the sum of token and positional embeddings.
        targets (ndarray):
            Ground truth token indices of shape (Batch, Time).
            Used for cross-entropy loss calculation.
        params (dict):
            Model parameters including weights for all N layers
            (W_q_i, W1_i, etc.) and hyperparameters.
        X_ids (ndarray):
            Original token indices of shape (Batch, Time).
            Stored in cache for embedding gradient computation.

    Returns:
        loss (float): Mean cross-entropy loss for the current batch.
        logits (ndarray): Raw output scores before softmax, shape (Batch, Time, Vocab).
        probs (ndarray): Probability distribution per token, shape (Batch, Time, Vocab).
        cache (dict):
            Intermediate values required for the backward pass,
            plus layer-specific caches and input metadata.
    """

    n_layers = params["n_layers"]
    W_out = params["W_out"]
    b_out = params["b_out"]

    # Storage for caches from each layer for the backward pass
    layer_caches = []

    # Process through each Transformer block sequentially
    current_h = X
    for i in range(n_layers):
        # We pass the layer index i so transformer_block knows which weights to use
        current_h, cache_block, _ = transformer_block(current_h, params, layer_idx=i)
        layer_caches.append(cache_block)

    # Final projection (current_h is the output of the last block)
    Y = current_h  # (B, T, D)
    logits = Y @ W_out + b_out  # (B, T, vocab)

    # Stable Softmax
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Cross-entropy Loss
    B, T = targets.shape
    b_idx = np.arange(B)[:, None]
    t_idx = np.arange(T)[None, :]
    logprobs = -np.log(probs[b_idx, t_idx, targets] + 1e-9)
    loss = np.mean(logprobs)

    # The cache now includes the list of layer_caches
    cache = {
        "X": X,  # Original input to the first block
        "Y": Y,  # Output of the last block
        "targets": targets,
        "probs": probs,
        "layer_caches": layer_caches,
        "W_out": W_out,
        "b_out": b_out,
        "X_ids": X_ids,
        "token_embed": params["token_embed"],
        "pos_embed": params["pos_embed"],
    }
    return loss, logits, probs, cache


def forward_no_loss(X, params):
    """
    Forward pass without targets or cross-entropy calculation.
    Used for inference and text generation.
    Returns: logits, probs, and the list of layer caches.
    """
    # Add a batch dimension if it's missing (B=1, T, D)
    if X.ndim == 2:
        X = X[np.newaxis, :, :]

    n_layers = params["n_layers"]
    W_out = params["W_out"]
    b_out = params["b_out"]

    layer_caches = []
    current_h = X

    # Process through each Transformer block sequentially
    for i in range(n_layers):
        # We pass layer_idx so the block knows which indexed weights to use
        current_h, cache_block, _ = transformer_block(current_h, params, layer_idx=i)
        layer_caches.append(cache_block)

    # Final linear projection onto vocabulary
    # Y shape: (1, T, D) -> logits shape: (1, T, vocab)
    Y = current_h
    logits = Y @ W_out + b_out

    # Stable Softmax for sampling
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Return the squeezed results for easier use in the generation loop
    # If B=1, we can return shapes (T, vocab)
    return logits[0], probs[0], layer_caches


def backward_pass(cache, params):
    """
    Orchestrates the full backward pass.
    From Loss -> Output Layer -> Transformer Block -> Embeddings.
    """
    # 1. Output Layer (Logits -> Y)
    probs = cache["probs"]
    targets = cache["targets"]
    Y = cache["Y"]
    B, T, D = Y.shape

    # Gradient of Cross-Entropy wrt Logits
    # dL/dz = (probs - targets) / N
    dlogits = probs.copy()
    dlogits[np.arange(B)[:, None], np.arange(T), targets] -= 1.0
    dlogits /= B * T

    # Gradients for W_out and b_out
    grads = {}
    grads["W_out"] = Y.reshape(B * T, D).T @ dlogits.reshape(B * T, -1)
    grads["b_out"] = np.sum(dlogits, axis=(0, 1))

    # Gradient flowing back to Y
    dY = dlogits @ params["W_out"].T  # (B, T, D)

    # 2. Transformer Block (Backward)
    # We unpack our block caches
    b_cache = cache["cache_block"]

    # Path: Y = X2 + mlp_out
    dmlp_out = dY
    dX2_res = dY

    # MLP Backward
    dX2_norm_mlp, dW1, db1, dW2, db2 = engine.backward_mlp(dmlp_out, b_cache["mlp"])
    grads.update({"W1": dW1, "b1": db1, "W2": dW2, "b2": db2})

    # LayerNorm 2 Backward
    dX2_ln = engine.backward_layernorm(dX2_norm_mlp, b_cache["ln2"])

    # Total dX2
    dX2 = dX2_res + dX2_ln

    # Path: X2 = X + att_out
    datt_out = dX2
    dX_res = dX2

    # Attention Backward
    dX_norm_att, dWq, dWk, dWv, dWo = engine.backward_attention(
        datt_out, b_cache["att"]
    )
    grads.update({"W_q": dWq, "W_k": dWk, "W_v": dWv, "W_o": dWo})

    # LayerNorm 1 Backward
    dX_ln = engine.backward_layernorm(dX_norm_att, b_cache["ln1"])

    # Total dX (Gradient at the entry of the Transformer block)
    dX_total = dX_res + dX_ln

    # 3. Embeddings Backward
    # We need X_ids from the cache to know where to send the gradients
    X_ids = cache["X_ids"]
    grads["token_embed"] = np.zeros_like(params["token_embed"])
    grads["pos_embed"] = np.zeros_like(params["pos_embed"])

    # Vectorized accumulation of gradients for embeddings
    np.add.at(grads["token_embed"], X_ids, dX_total)
    # For positions, we sum across the batch dimension
    grads["pos_embed"] += np.sum(dX_total, axis=0)

    return grads


def output_layer_backward(cache):
    """
    Performs the backward pass for the final linear layer and cross-entropy loss.

    This function computes the gradient of the loss with respect to the output
    logits, then backpropagates it to calculate gradients for the output weights
    (W_out, b_out) and the upstream hidden states (Y).

    Args:
        cache (dict): A dictionary containing:
            - "logits" (ndarray): Raw scores before softmax, shape (Batch*Time, Vocab).
            - "probs" (ndarray): Softmax probabilities, shape (Batch*Time, Vocab).
            - "targets" (ndarray): Ground truth token indices, shape (Batch*Time,).
            - "W_out" (ndarray): Final projection matrix, shape (d_model, Vocab).
            - "Y" (ndarray): Input hidden states to this layer, shape (Batch*Time, d_model).

    Returns:
        tuple: (dW_out, db_out, dY)
            - dW_out: Gradient with respect to output weights.
            - db_out: Gradient with respect to output bias.
            - dY: Gradient with respect to the input hidden states (to be sent to Transformer blocks).
    """

    # logits: (seq_len, V), probs: (seq_len, V)
    logits = cache["logits"]
    probs = cache["probs"]
    targets = cache["targets"]
    seq_len = logits.shape[0]
    W_out = cache["W_out"]

    # gradient of mean cross-entropy wrt logits:
    dlogits = probs.copy()  # (seq_len, V)
    dlogits[np.arange(seq_len), targets] -= 1.0
    dlogits /= seq_len  # because loss is mean over positions

    # grads for W_out, b_out and upstream to Y
    Y = cache["Y"]  # (seq_len, d_model)
    # W_out: (d_model, V)
    dW_out = Y.T @ dlogits  # (d_model, V)
    db_out = np.sum(dlogits, axis=0)  # (V,)

    dY = dlogits @ W_out.T  # (seq_len, d_model)

    return dW_out, db_out, dY


def generate_stream(prompt, vocab, params, max_new_tokens=512, k=20, temperature=0.8):
    """
    Generates text token by token (streaming) using top-k sampling and temperature.
    yields: Each newly generated character.
    """
    # 1) Encode prompt to token IDs
    ids = utils.encode(prompt, vocab)
    vocab_size = len(vocab)
    block_size = params["block_size"]
    token_embed = params["token_embed"]
    pos_embed = params["pos_embed"]

    for _ in range(max_new_tokens):
        # 2) Crop context to the last 'block_size' tokens
        context_ids = ids[-block_size:]

        # 3) Create embeddings for the current context
        tok_emb = token_embed[context_ids]
        pos_emb_block = pos_embed[np.arange(len(context_ids))]
        X = tok_emb + pos_emb_block

        # 4) Forward pass without loss
        _, probs, _ = forward_no_loss(X, params)

        # 5) Get probabilities of the very last token
        p = probs[-1].astype(np.float64)

        # 6) Apply Temperature scaling
        if temperature != 1.0:
            p = np.exp(np.log(p + 1e-10) / temperature)
            p /= p.sum()

        # 7) Top-k sampling
        if k is not None and k > 0:
            actual_k = min(k, vocab_size)
            top_idx = np.argpartition(p, -actual_k)[-actual_k:]
            top_probs = p[top_idx]
            top_probs /= top_probs.sum() + 1e-12
            next_id = np.random.choice(top_idx, p=top_probs)
        else:
            p /= p.sum() + 1e-12
            next_id = np.random.choice(vocab_size, p=p)

        # 8) Update sequence and yield the new character
        ids.append(int(next_id))

        # Decode only the last token to stream it
        yield utils.decode([next_id], vocab)
