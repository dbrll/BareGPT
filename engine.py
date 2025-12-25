import numpy as np

"""
BareGPT Engine: The computational heart of the backpropagation.
This file contains the manual implementation of gradients for every layer.
"""


def adam_init(params):
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    return m, v


def adam_step(
    params, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8, clip_norm=None
):
    """
    Performs a standard Adam update step with gradient clipping.

    Adam uses adaptive learning rates for each parameter, mimicking momentum.
    It converges much faster than basic SGD by scaling updates based on historical gradients.

    Args:
        params, grads: dicts mapping parameter names to numpy arrays
        m, v: dicts storing first and second moments
        t: current timestep (starting from 1)
        lr: learning rate
        clip_norm: if provided, the global gradient norm will be clipped to this value

    Returns:
        params, m, v, last_update: updated dictionaries and the last computed update
    """
    # 1. Global Gradient Clipping
    if clip_norm is not None:
        total_norm = 0.0
        for g in grads.values():
            total_norm += np.sum(g * g)
        total_norm = np.sqrt(total_norm)

        if total_norm > clip_norm:
            scale = clip_norm / (total_norm + 1e-12)
            for k in grads:
                grads[k] = grads[k] * scale

    # 2. Parameter Updates
    updates = {}
    for k in params.keys():
        # Skip hyperparameters stored in params (like n_heads) that don't have gradients
        g = grads.get(k)
        if g is None:
            continue

        # Initialize m/v if they don't exist or have wrong shapes
        if k not in m or m[k].shape != params[k].shape:
            m[k] = np.zeros_like(params[k])
        if k not in v or v[k].shape != params[k].shape:
            v[k] = np.zeros_like(params[k])

        # Update biased moments
        m[k] = beta1 * m[k] + (1 - beta1) * g
        v[k] = beta2 * v[k] + (1 - beta2) * (g * g)

        # Bias correction
        m_hat = m[k] / (1 - beta1**t)
        v_hat = v[k] / (1 - beta2**t)

        # Apply update
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        params[k] -= update
        updates[k] = update

    return params, m, v, updates


def backward_and_get_grads(cache):
    """
    Performs the full backward pass through all Transformer layers and embeddings.

    The process follows the 'Chain Rule' in reverse order:
    1. Output Head: Gradients for W_out and b_out.
    2. Transformer Blocks: Iterates from the last layer (N-1) down to the first (0).
    3. Embeddings: Gradients for token and positional embeddings.

    Args:
        cache (dict): Global cache containing intermediate activations from the forward pass,
                      layer-specific caches, and input metadata (X_ids)

    Returns:
        grads (dict): A dictionary containing gradients for every trainable parameter
                      indexed by name (e.g., 'W_q_0', 'token_embed')
    """
    # ---- 1) Extraction from the global cache ----
    Y = cache["Y"]
    targets = cache["targets"]
    probs = cache["probs"]
    W_out = cache["W_out"]
    X_ids = cache["X_ids"]
    token_embed = cache["token_embed"]
    pos_embed = cache["pos_embed"]
    layer_caches = cache["layer_caches"]

    B, T, D = Y.shape
    V = probs.shape[-1]
    n_layer = len(layer_caches)

    # ---- 2) Output Layer Gradients ----
    # dlogits = (probs - one_hot) / (B * T)
    dlogits = probs.copy()
    dlogits[np.arange(B)[:, None], np.arange(T), targets] -= 1.0
    dlogits /= B * T

    # Reshape for matrix multiplication: (D, V)
    dW_out = Y.reshape(B * T, D).T @ dlogits.reshape(B * T, V)
    db_out = np.sum(dlogits, axis=(0, 1))

    # Gradient flows back to the hidden states: (B, T, D)
    d_current = dlogits @ W_out.T

    grads = {
        "W_out": dW_out,
        "b_out": db_out,
    }

    # ---- 3) Transformer Blocks (Reverse Cascade) ----
    for i in reversed(range(n_layer)):
        layer_cache = layer_caches[i]

        (
            d_current,  # Gradient flows from layer i to layer i-1
            dW_q,
            dW_k,
            dW_v,
            dW_o,
            dW1,
            db1,
            dW2,
            db2,
            (dgamma1, dbeta1),
            (dgamma2, dbeta2),
        ) = backward_transformer_block(d_current, layer_cache)

        # Update global grads dictionary with indexed keys
        grads.update(
            {
                f"W_q_{i}": dW_q,
                f"W_k_{i}": dW_k,
                f"W_v_{i}": dW_v,
                f"W_o_{i}": dW_o,
                f"W1_{i}": dW1,
                f"b1_{i}": db1,
                f"W2_{i}": dW2,
                f"b2_{i}": db2,
                f"ln1_dgamma_{i}": dgamma1,
                f"ln1_dbeta_{i}": dbeta1,
                f"ln2_dgamma_{i}": dgamma2,
                f"ln2_dbeta_{i}": dbeta2,
            }
        )

    # ---- 4) Embeddings Gradient (Vectorized Optimization) ----
    d_token_embed = np.zeros_like(token_embed)
    d_pos_embed = np.zeros_like(pos_embed)

    # Optimization: Use np.add.at to avoid slow Python loops.
    # It adds d_current values into d_token_embed at indices specified by X_ids.
    np.add.at(d_token_embed, X_ids, d_current)
    """
    for b in range(B):
        for t in range(T):
            tok = X_ids[b, t]
            d_token_embed[tok] += d_current[b, t]  # Gradient flows into tokens
            d_pos_embed[t] += d_current[b, t]  # Gradient flows into positions
    """

    # For positional embeddings, we sum across the batch dimension
    # since the same position embedding is added to every sequence in the batch.
    d_pos_embed = np.sum(d_current, axis=0)

    grads["token_embed"] = d_token_embed
    grads["pos_embed"] = d_pos_embed

    return grads


def backward_layernorm(dout, x, gamma=1, beta=1, eps=1e-5):
    """
    Backward pass for Layer Normalization.
    Mathematical derivation involves the chain rule over mean and variance.
    """

    B, T, D = x.shape

    # ---- Forward stats (recomputées) ----
    mean = np.mean(x, axis=-1, keepdims=True)  # (B,T,1)
    var = np.var(x, axis=-1, keepdims=True)  # (B,T,1)
    std = np.sqrt(var + eps)  # (B,T,1)
    x_hat = (x - mean) / std  # (B,T,D)

    # ---- Backward ----
    dxhat = dout * gamma  # (B,T,D)

    dvar = np.sum(
        dxhat * (x - mean) * -0.5 * (std**-3), axis=-1, keepdims=True
    )  # (B,T,1)

    dmean = np.sum(-dxhat / std, axis=-1, keepdims=True) + dvar * np.mean(
        -2 * (x - mean), axis=-1, keepdims=True
    )  # (B,T,1)

    dx = dxhat / std + dvar * 2 * (x - mean) / D + dmean / D  # (B,T,D)

    # ---- Grad gamma / beta ----
    dgamma = np.sum(dout * x_hat, axis=(0, 1))  # (D,)
    dbeta = np.sum(dout, axis=(0, 1))  # (D,)

    return dx, dgamma, dbeta


def backward_transformer_block(dY, cache_block):
    """
    Performs the backward pass for a single Transformer block.

    The gradient dY flows backwards through the MLP sub-layer, then the
    Attention sub-layer, accumulating gradients for weights and handling
    residual connections at each step.

    Args:
        dY (ndarray):
            Gradient of the loss with respect to the block output Y, shape (B, T, D).
        cache_block (dict):
            Dictionary containing intermediate activations and caches
            for Attention, MLP, and LayerNorms.

    Returns:
        tuple: (dX_block, dW_q, dW_k, dW_v, dW_o, dW1, db1, dW2, db2, layer_norms)
               Where dX_block is the gradient to be passed to the previous layer.
    """

    # --- 1. Unpack Cache ---
    ln1_cache = cache_block["ln1"]
    att_cache = cache_block["att"]
    ln2_cache = cache_block["ln2"]
    mlp_cache = cache_block["mlp"]

    # --- 2. Feed-Forward (MLP) Sub-layer Backward ---
    # Y = X2 + mlp_out
    # The gradient dY splits: one part goes to the residual, one to the MLP
    dmlp_out = dY.copy()
    dX2_from_residual = dY.copy()

    # Backprop through MLP: returns gradient wrt MLP input and weights
    dX2_norm_from_mlp, dW1, db1, dW2, db2 = backward_mlp(dmlp_out, mlp_cache)

    # Backprop through second LayerNorm
    eps2 = ln2_cache.get("eps", 1e-5) if isinstance(ln2_cache, dict) else 1e-5
    dX2_from_ln2, dgamma2, dbeta2 = backward_layernorm(
        dX2_norm_from_mlp, ln2_cache["x"], 1.0, eps=eps2
    )

    # Total gradient at X2 (sum of residual and path through MLP)
    dX2_total = dX2_from_residual + dX2_from_ln2

    # --- 3. Self-Attention Sub-layer Backward ---
    # X2 = X + att_matrix
    # Again, the gradient splits between the residual and the attention mechanism
    dX_from_residual1 = dX2_total.copy()
    d_att_matrix = dX2_total.copy()

    # Backprop through Attention: returns gradient wrt input and projection weights
    dX_norm1_from_att, dW_q, dW_k, dW_v, dW_o = backward_attention(
        d_att_matrix, att_cache
    )

    # Backprop through first LayerNorm
    eps1 = ln1_cache.get("eps", 1e-5) if isinstance(ln1_cache, dict) else 1e-5
    dX_from_ln1, dgamma1, dbeta1 = backward_layernorm(
        dX_norm1_from_att, ln1_cache["x"], 1.0, eps=eps1
    )

    # --- 4. Final Gradient for the Block ---
    # Total gradient on X to be passed to the previous block or embeddings
    dX_block = dX_from_residual1 + dX_from_ln1

    return (
        dX_block,
        dW_q,
        dW_k,
        dW_v,
        dW_o,
        dW1,
        db1,
        dW2,
        db2,
        (dgamma1, dbeta1),
        (dgamma2, dbeta2),
    )


def backward_mlp(dY, cache):
    """
    Backward pass for the MLP block.
    Derives through Linear -> GELU -> Linear.
    """
    X, Z1, A1, W1, W2 = cache["X"], cache["Z1"], cache["A1"], cache["W1"], cache["W2"]
    B, T, D = X.shape

    # 1. Backprop through second linear layer
    # dZ2 = dY (since Z2 was the output)
    # dW2 = A1.reshape(-1, A1.shape[-1]).T @ dY.reshape(-1, dY.shape[-1])
    # ^ NumPy has a fancier way to express this through Einstein summation:
    dW2 = np.einsum("bti,btj->ij", A1, dY)
    db2 = np.sum(dY, axis=(0, 1))
    dA1 = dY @ W2.T

    # 2. Backprop through GELU activation
    # Derivative of tanh-approximation of GELU
    s = np.sqrt(2 / np.pi)
    # This is the local gradient of GELU(Z1)
    tanh_part = np.tanh(s * (Z1 + 0.044715 * Z1**3))
    d_gelu = 0.5 * (1.0 + tanh_part) + 0.5 * Z1 * (1.0 - tanh_part**2) * s * (
        1.0 + 3 * 0.044715 * Z1**2
    )
    dZ1 = dA1 * d_gelu

    # 3. Backprop through first linear layer
    # dW1 = X.reshape(-1, X.shape[-1]).T @ dZ1.reshape(-1, dZ1.shape[-1])
    # Like for dW2, we'll use an Einstein summation instead:
    dW1 = np.einsum("bti,btj->ij", X, dZ1)
    db1 = np.sum(dZ1, axis=(0, 1))
    dX = dZ1 @ W1.T

    return dX, dW1, db1, dW2, db2


def backward_attention(d_att_out, cache):
    """
    Backward pass for Multi-Head Masked Self-Attention.

    This function computes gradients for the Query, Key, Value, and Output projections,
    as well as the gradient flowing back to the input hidden states (dX).
    This is the most complex piece of the transformer.

    Args:
        d_att_out (ndarray): Gradient of the output of attention, shape (B, T, D).
        cache (dict): Intermediate values from forward pass (Qh, Kh, Vh, att_probs, etc.).

    Returns:
        tuple: (dX, dW_q, dW_k, dW_v, dW_o)
    """
    # 1. Unpack cache
    X = cache["X"]
    Qh, Kh, Vh = cache["Qh"], cache["Kh"], cache["Vh"]
    att_probs = cache["att_probs"]
    W_q, W_k, W_v, W_o = cache["W_q"], cache["W_k"], cache["W_v"], cache["W_o"]

    B, T, D = X.shape
    H = cache["n_head"]
    Hd = D // H

    # 2. Backprop through Output Projection (W_o) and Concatenation
    # att_out = context_flat @ W_o
    # dW_o = context_flat.T @ d_att_out
    d_context_flat = d_att_out @ W_o.T  # (B, T, D)

    # Reconstructing context_flat for dW_o calculation
    # att_out_h = att_probs @ Vh
    att_out_h = att_probs @ Vh  # (B, H, T, Hd)
    context_flat = att_out_h.transpose(0, 2, 1, 3).reshape(B, T, D)

    # Flatten B and T to compute weight gradients: (D, BT) @ (BT, D) -> (D, D)
    # dW_o = np.einsum("bti,btj->ij", context_flat, d_att_out)
    dW_o = context_flat.reshape(-1, D).T @ d_att_out.reshape(-1, D)

    # 3. Backprop through Multi-Head Split
    # Reshape d_context_flat back to (B, H, T, Hd)
    d_att_out_h = d_context_flat.reshape(B, T, H, Hd).transpose(0, 2, 1, 3)

    # 4. Backprop through: att_probs @ Vh
    # d_att_probs = d_att_out_h @ Vh^T
    # dVh = att_probs^T @ d_att_out_h
    # d_att_probs = np.einsum("bhid,bhjd->bhij", d_att_out_h, Vh)
    d_att_probs = d_att_out_h @ Vh.transpose(0, 1, 3, 2)

    # dVh = np.einsum("bhij,bhid->bhjd", att_probs, d_att_out_h)
    dVh = att_probs.transpose(0, 1, 3, 2) @ d_att_out_h

    # 5. Backprop through Softmax
    # Formula: d_scores = P * (dP - sum(dP * P, axis=-1))
    d_scores = att_probs * (
        d_att_probs - np.sum(d_att_probs * att_probs, axis=-1, keepdims=True)
    )

    # 6. Backprop through scaling (1/sqrt(Hd)) and dot product (Qh @ Kh^T)
    d_scores /= np.sqrt(Hd)

    # dQh = d_scores @ Kh
    # dKh = d_scores^T @ Qh
    # dQh = np.einsum("bhij,bhjd->bhid", d_scores, Kh)
    dQh = d_scores @ Kh

    # dKh = np.einsum("bhij,bhid->bhjd", d_scores, Qh)
    dKh = d_scores.transpose(0, 1, 3, 2) @ Qh

    # 7. Reshape and project gradients back to original Q, K, V dimensions (B, T, D)
    dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T, D)
    dK = dKh.transpose(0, 2, 1, 3).reshape(B, T, D)
    dV = dVh.transpose(0, 2, 1, 3).reshape(B, T, D)

    # 8. Gradients for W_q, W_k, W_v
    # Flatten X and dQ/dK/dV to compute gradients: (D, BT) @ (BT, D)
    X_flat = X.reshape(-1, D)
    dW_q = X_flat.T @ dQ.reshape(-1, D)
    dW_k = X_flat.T @ dK.reshape(-1, D)
    dW_v = X_flat.T @ dV.reshape(-1, D)

    # 9. Gradient flowing back to input X
    dX = dQ @ W_q.T + dK @ W_k.T + dV @ W_v.T

    return dX, dW_q, dW_k, dW_v, dW_o
