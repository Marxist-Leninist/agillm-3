"""
NAT-Mamba: Non-Autoregressive Mamba via Bidirectional SSM Encoder-Decoder

Mathematical foundation:
- Standard Mamba: h_t = A_t * h_{t-1} + B_t * x_t (inherently causal)
- NAT-Mamba: Encode bidirectionally with SSM, decode in parallel

Architecture:
1. Bidirectional SSM Encoder: Forward + Backward scans capture full context
2. Attention Pooling: Aggregate into position-aware global representations  
3. Parallel Decoder: Generate all outputs simultaneously from global state + position

This preserves SSM's recurrent inductive bias for encoding while enabling
parallel decoding - the "right" way to do NAT with state space models.

Author: Scott + Claude collaborative research session
Date: 2024
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


class NATMamba:
    """
    Non-Autoregressive Mamba model.
    
    Uses bidirectional SSM encoding with parallel decoding.
    O(T) encoding (with parallel scan), O(T²) pooling attention, O(T) decoding.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Args:
            vocab_size: Size of token vocabulary
            d_model: Model dimension
            d_state: SSM state dimension
            n_layers: Number of encoder layers
            max_seq_len: Maximum sequence length
            dropout: Dropout rate (for training)
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        if seed is not None:
            np.random.seed(seed)
        
        self._init_params()
        self.cache: Dict[str, Any] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.training = True
        
    def _init_params(self):
        """Xavier/He initialization for all parameters."""
        d, s, v = self.d_model, self.d_state, self.vocab_size
        
        # Embeddings
        self.embed = np.random.randn(v, d) * np.sqrt(2/d)
        self.pos_embed = np.random.randn(self.max_seq_len, d) * 0.02
        
        # Per-layer SSM parameters (forward and backward)
        self.layers = []
        for _ in range(self.n_layers):
            layer = {
                # Forward SSM
                'W_a_fwd': np.random.randn(d, s) * np.sqrt(2/d),
                'W_b_fwd': np.random.randn(d, s) * np.sqrt(2/d),
                'W_c_fwd': np.random.randn(d, s) * np.sqrt(2/d),
                # Backward SSM
                'W_a_bwd': np.random.randn(d, s) * np.sqrt(2/d),
                'W_b_bwd': np.random.randn(d, s) * np.sqrt(2/d),
                'W_c_bwd': np.random.randn(d, s) * np.sqrt(2/d),
                # Output projection (combines fwd + bwd)
                'W_out': np.random.randn(s * 2, d) * np.sqrt(1/s),
                # Layer norm params
                'ln_gamma': np.ones(d),
                'ln_beta': np.zeros(d),
            }
            self.layers.append(layer)
        
        # Attention pooling
        self.W_pool_q = np.random.randn(d, s) * np.sqrt(2/d)
        self.W_pool_k = np.random.randn(d, s) * np.sqrt(2/d)
        self.W_pool_v = np.random.randn(d, s) * np.sqrt(2/d)
        
        # Decoder
        self.W_decode = np.random.randn(s + d, d) * np.sqrt(2/(s + d))
        self.W_vocab = np.random.randn(d, v) * np.sqrt(2/d)
        
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                    eps: float = 1e-5) -> Tuple[np.ndarray, Dict]:
        """Layer normalization with cache for backward pass."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        out = gamma * x_norm + beta
        cache = {'x': x, 'x_norm': x_norm, 'mean': mean, 'var': var, 'gamma': gamma}
        return out, cache
    
    def _ssm_scan(
        self, 
        x: np.ndarray, 
        W_a: np.ndarray, 
        W_b: np.ndarray, 
        W_c: np.ndarray,
        reverse: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Selective SSM scan (sequential version).
        
        For production, replace with parallel associative scan.
        
        Args:
            x: Input tensor (B, T, d_model)
            W_a, W_b, W_c: Projection matrices
            reverse: If True, scan backwards
            
        Returns:
            y: Output tensor (B, T, d_state)
            cache: Cached values for backward pass
        """
        B, T, D = x.shape
        
        # Compute selective parameters
        log_a = -np.abs(x @ W_a)  # Negative for decay/stability
        b = x @ W_b
        c = x @ W_c
        
        if reverse:
            log_a = log_a[:, ::-1, :]
            b = b[:, ::-1, :]
            c = c[:, ::-1, :]
        
        # Sequential scan
        h_states = np.zeros((B, T, self.d_state))
        h = np.zeros((B, self.d_state))
        
        for t in range(T):
            a_t = np.exp(log_a[:, t, :])
            h = a_t * h + b[:, t, :]
            h_states[:, t, :] = h
        
        # Output
        y = c * h_states
        
        if reverse:
            y = y[:, ::-1, :]
            h_states = h_states[:, ::-1, :]
        
        cache = {
            'x': x, 'log_a': log_a, 'b': b, 'c': c, 
            'h_states': h_states, 'reverse': reverse
        }
        
        return y, cache
    
    def _attention_pool(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Attention-based pooling to create position-aware global context.
        
        Each position attends to all positions to get its context.
        """
        B, T, D = x.shape
        
        q = x @ self.W_pool_q
        k = x @ self.W_pool_k
        v = x @ self.W_pool_v
        
        # Scaled dot-product attention
        scores = np.einsum('btd,bsd->bts', q, k) / np.sqrt(self.d_state)
        
        # Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        
        # Weighted sum
        out = np.einsum('bts,bsd->btd', attn, v)
        
        cache = {'x': x, 'q': q, 'k': k, 'v': v, 'attn': attn}
        return out, cache
    
    def forward(
        self, 
        tokens: np.ndarray, 
        targets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """
        Forward pass.
        
        Args:
            tokens: Input token IDs (B, T)
            targets: Target token IDs for loss computation (B, T)
            
        Returns:
            logits: Output logits (B, T, vocab_size)
            probs: Output probabilities (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        B, T = tokens.shape
        self.cache['T'] = T
        self.cache['tokens'] = tokens
        
        # Embed tokens + positions
        x = self.embed[tokens] + self.pos_embed[:T]
        self.cache['x_input'] = x
        
        # Encoder layers
        self.cache['layer_caches'] = []
        for i, layer in enumerate(self.layers):
            # Layer norm
            x_ln, ln_cache = self._layer_norm(x, layer['ln_gamma'], layer['ln_beta'])
            
            # Bidirectional SSM
            y_fwd, fwd_cache = self._ssm_scan(
                x_ln, layer['W_a_fwd'], layer['W_b_fwd'], layer['W_c_fwd'], 
                reverse=False
            )
            y_bwd, bwd_cache = self._ssm_scan(
                x_ln, layer['W_a_bwd'], layer['W_b_bwd'], layer['W_c_bwd'],
                reverse=True
            )
            
            # Combine and project
            y_bi = np.concatenate([y_fwd, y_bwd], axis=-1)
            y_out = y_bi @ layer['W_out']
            
            # Residual
            x = x + y_out
            
            self.cache['layer_caches'].append({
                'ln_cache': ln_cache,
                'fwd_cache': fwd_cache,
                'bwd_cache': bwd_cache,
                'y_bi': y_bi,
                'x_out': x
            })
        
        self.cache['encoder_out'] = x
        
        # Attention pooling
        global_ctx, pool_cache = self._attention_pool(x)
        self.cache['pool_cache'] = pool_cache
        self.cache['global_ctx'] = global_ctx
        
        # Decoder: combine global context with position info
        pos_broadcast = np.broadcast_to(self.pos_embed[:T], (B, T, self.d_model))
        decode_input = np.concatenate([global_ctx, pos_broadcast], axis=-1)
        self.cache['decode_input'] = decode_input
        
        h_out = np.tanh(decode_input @ self.W_decode)
        self.cache['h_out'] = h_out
        
        # Output logits
        logits = h_out @ self.W_vocab
        
        # Softmax
        logits_max = logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        self.cache['probs'] = probs
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            target_probs = probs[np.arange(B)[:, None], np.arange(T), targets]
            loss = -np.log(target_probs + 1e-10).mean()
            self.cache['targets'] = targets
        
        return logits, probs, loss
    
    def backward(self) -> Dict[str, np.ndarray]:
        """
        Backward pass. Must call forward() with targets first.
        
        Returns gradients for all parameters.
        """
        c = self.cache
        B, T = c['targets'].shape
        
        # Output layer gradients
        d_logits = c['probs'].copy()
        d_logits[np.arange(B)[:, None], np.arange(T), c['targets']] -= 1
        d_logits /= (B * T)
        
        self.grads['W_vocab'] = np.einsum('btd,btv->dv', c['h_out'], d_logits)
        d_h_out = d_logits @ self.W_vocab.T
        
        # Tanh backward: d/dx tanh(x) = 1 - tanh²(x)
        # h_out = tanh(decode_input @ W_decode)
        d_pre_tanh = d_h_out * (1 - c['h_out']**2)  # (B, T, d_model)
        self.grads['W_decode'] = np.einsum('btd,btm->dm', c['decode_input'], d_pre_tanh)
        d_decode_input = d_pre_tanh @ self.W_decode.T  # (B, T, d_state + d_model)
        
        # Split gradient for global_ctx and pos_embed
        # decode_input = [global_ctx (d_state), pos_embed (d_model)]
        d_global_ctx = d_decode_input[:, :, :self.d_state]
        d_pos_from_decode = d_decode_input[:, :, self.d_state:].sum(axis=0)  # (T, d_model)
        
        # Attention pooling backward
        pc = c['pool_cache']
        d_attn = np.einsum('btd,bsd->bts', d_global_ctx, pc['v'])
        d_v = np.einsum('bts,btd->bsd', pc['attn'], d_global_ctx)
        
        d_scores = pc['attn'] * (d_attn - (d_attn * pc['attn']).sum(axis=-1, keepdims=True))
        d_scores /= np.sqrt(self.d_state)
        
        d_q = np.einsum('bts,bsd->btd', d_scores, pc['k'])
        d_k = np.einsum('bts,btd->bsd', d_scores, pc['q'])
        
        self.grads['W_pool_q'] = np.einsum('btd,bts->ds', pc['x'], d_q)
        self.grads['W_pool_k'] = np.einsum('btd,bts->ds', pc['x'], d_k)
        self.grads['W_pool_v'] = np.einsum('btd,bts->ds', pc['x'], d_v)
        
        d_encoder_out = (d_q @ self.W_pool_q.T + d_k @ self.W_pool_k.T + 
                         d_v @ self.W_pool_v.T)
        
        # Backward through encoder layers (reverse order)
        d_x = d_encoder_out
        self.grads['layers'] = []
        
        for i in range(self.n_layers - 1, -1, -1):
            layer = self.layers[i]
            lc = c['layer_caches'][i]
            
            layer_grads = {}
            
            # Residual
            d_y_out = d_x
            d_residual = d_x
            
            # W_out backward
            layer_grads['W_out'] = np.einsum('btd,btm->dm', lc['y_bi'], d_y_out)
            d_y_bi = d_y_out @ layer['W_out'].T
            
            d_y_fwd = d_y_bi[:, :, :self.d_state]
            d_y_bwd = d_y_bi[:, :, self.d_state:]
            
            # SSM backward (simplified - gradient through projections)
            fc = lc['fwd_cache']
            bc = lc['bwd_cache']
            
            # Forward SSM gradients
            d_c_fwd = d_y_fwd * fc['h_states']
            layer_grads['W_c_fwd'] = np.einsum('btd,bts->ds', fc['x'], d_c_fwd)
            layer_grads['W_b_fwd'] = np.einsum('btd,bts->ds', fc['x'], d_y_fwd)
            layer_grads['W_a_fwd'] = np.einsum('btd,bts->ds', fc['x'], d_y_fwd * 0.1)
            
            # Backward SSM gradients  
            d_c_bwd = d_y_bwd * bc['h_states']
            layer_grads['W_c_bwd'] = np.einsum('btd,bts->ds', bc['x'], d_c_bwd)
            layer_grads['W_b_bwd'] = np.einsum('btd,bts->ds', bc['x'], d_y_bwd)
            layer_grads['W_a_bwd'] = np.einsum('btd,bts->ds', bc['x'], d_y_bwd * 0.1)
            
            # Gradient through layer norm (simplified)
            lnc = lc['ln_cache']
            d_x_ln = (d_y_fwd @ layer['W_b_fwd'].T + d_y_bwd @ layer['W_b_bwd'].T)
            
            layer_grads['ln_gamma'] = (d_x_ln * lnc['x_norm']).sum(axis=(0, 1))
            layer_grads['ln_beta'] = d_x_ln.sum(axis=(0, 1))
            
            d_x = d_residual + d_x_ln * lnc['gamma'] / np.sqrt(lnc['var'] + 1e-5)
            
            self.grads['layers'].insert(0, layer_grads)
        
        # Embedding gradients
        self.grads['embed'] = np.zeros_like(self.embed)
        np.add.at(self.grads['embed'], c['tokens'], d_x)
        
        self.grads['pos_embed'] = np.zeros_like(self.pos_embed)
        self.grads['pos_embed'][:T] = d_x.sum(axis=0) + d_pos_from_decode
        
        return self.grads
    
    def step(self, lr: float, weight_decay: float = 0.0):
        """
        SGD update step with optional weight decay.
        
        Args:
            lr: Learning rate
            weight_decay: L2 regularization coefficient
        """
        T = self.cache['T']
        
        # Global params
        self.embed -= lr * (self.grads['embed'] + weight_decay * self.embed)
        self.pos_embed[:T] -= lr * self.grads['pos_embed'][:T]
        self.W_pool_q -= lr * (self.grads['W_pool_q'] + weight_decay * self.W_pool_q)
        self.W_pool_k -= lr * (self.grads['W_pool_k'] + weight_decay * self.W_pool_k)
        self.W_pool_v -= lr * (self.grads['W_pool_v'] + weight_decay * self.W_pool_v)
        self.W_decode -= lr * (self.grads['W_decode'] + weight_decay * self.W_decode)
        self.W_vocab -= lr * (self.grads['W_vocab'] + weight_decay * self.W_vocab)
        
        # Per-layer params
        for i, layer in enumerate(self.layers):
            lg = self.grads['layers'][i]
            for key in lg:
                layer[key] -= lr * (lg[key] + weight_decay * layer[key])
    
    def generate(self, tokens: np.ndarray) -> np.ndarray:
        """
        Generate output tokens (parallel, non-autoregressive).
        
        Args:
            tokens: Input tokens (B, T)
            
        Returns:
            predicted: Predicted token IDs (B, T)
        """
        self.training = False
        _, probs, _ = self.forward(tokens)
        self.training = True
        return probs.argmax(axis=-1)
    
    def save(self, path: str):
        """Save model parameters to npz file."""
        params = {
            'embed': self.embed,
            'pos_embed': self.pos_embed,
            'W_pool_q': self.W_pool_q,
            'W_pool_k': self.W_pool_k,
            'W_pool_v': self.W_pool_v,
            'W_decode': self.W_decode,
            'W_vocab': self.W_vocab,
        }
        # Add layer params
        for i, layer in enumerate(self.layers):
            for key, val in layer.items():
                params[f'layer_{i}_{key}'] = val
        
        # Add config
        params['config'] = np.array([
            self.vocab_size, self.d_model, self.d_state, 
            self.n_layers, self.max_seq_len
        ])
        
        np.savez(path, **params)
    
    @classmethod
    def load(cls, path: str) -> 'NATMamba':
        """Load model from npz file."""
        data = np.load(path)
        config = data['config']
        
        model = cls(
            vocab_size=int(config[0]),
            d_model=int(config[1]),
            d_state=int(config[2]),
            n_layers=int(config[3]),
            max_seq_len=int(config[4])
        )
        
        model.embed = data['embed']
        model.pos_embed = data['pos_embed']
        model.W_pool_q = data['W_pool_q']
        model.W_pool_k = data['W_pool_k']
        model.W_pool_v = data['W_pool_v']
        model.W_decode = data['W_decode']
        model.W_vocab = data['W_vocab']
        
        for i in range(model.n_layers):
            for key in model.layers[i]:
                model.layers[i][key] = data[f'layer_{i}_{key}']
        
        return model


# =============================================================================
# Training utilities
# =============================================================================

class NATMambaTrainer:
    """Training loop with logging and checkpointing."""
    
    def __init__(
        self,
        model: NATMamba,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        lr_decay: float = 1.0,
        clip_grad: Optional[float] = 1.0
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.clip_grad = clip_grad
        self.step_count = 0
        
    def _clip_gradients(self):
        """Clip gradients by global norm."""
        if self.clip_grad is None:
            return
        
        total_norm = 0.0
        for key, grad in self.model.grads.items():
            if key == 'layers':
                for lg in grad:
                    for g in lg.values():
                        total_norm += (g ** 2).sum()
            else:
                total_norm += (grad ** 2).sum()
        
        total_norm = np.sqrt(total_norm)
        
        if total_norm > self.clip_grad:
            scale = self.clip_grad / total_norm
            for key, grad in self.model.grads.items():
                if key == 'layers':
                    for lg in grad:
                        for k in lg:
                            lg[k] *= scale
                else:
                    self.model.grads[key] *= scale
    
    def train_step(self, tokens: np.ndarray, targets: np.ndarray) -> float:
        """Single training step."""
        _, _, loss = self.model.forward(tokens, targets)
        self.model.backward()
        self._clip_gradients()
        
        current_lr = self.lr * (self.lr_decay ** self.step_count)
        self.model.step(current_lr, self.weight_decay)
        
        self.step_count += 1
        return loss
    
    def evaluate(self, tokens: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
        """Evaluate loss and accuracy."""
        _, probs, loss = self.model.forward(tokens, targets)
        preds = probs.argmax(axis=-1)
        acc = (preds == targets).mean()
        return loss, acc


# =============================================================================
# Demo / Test
# =============================================================================

def demo():
    """Demo: train on reverse sequence task."""
    print("=" * 60)
    print("NAT-Mamba Demo: Reverse Sequence Task")
    print("=" * 60)
    
    # Config
    vocab_size = 16
    d_model = 64
    d_state = 32
    n_layers = 2
    seq_len = 8
    batch_size = 64
    n_steps = 300
    
    # Model
    model = NATMamba(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        max_seq_len=seq_len,
        seed=42
    )
    
    trainer = NATMambaTrainer(model, lr=0.05, clip_grad=1.0)
    
    print(f"\nConfig: vocab={vocab_size}, d_model={d_model}, d_state={d_state}")
    print(f"        n_layers={n_layers}, seq_len={seq_len}")
    print(f"        batch_size={batch_size}, steps={n_steps}\n")
    
    # Training loop
    for step in range(n_steps):
        # Generate batch: random sequences
        tokens = np.random.randint(0, vocab_size, (batch_size, seq_len))
        targets = tokens[:, ::-1]  # Reverse
        
        loss = trainer.train_step(tokens, targets)
        
        if step % 50 == 0:
            _, acc = trainer.evaluate(tokens, targets)
            print(f"Step {step:4d}: loss={loss:.4f}, acc={acc:.3f}")
    
    # Final evaluation
    tokens = np.random.randint(0, vocab_size, (batch_size, seq_len))
    targets = tokens[:, ::-1]
    loss, acc = trainer.evaluate(tokens, targets)
    
    print(f"\nFinal: loss={loss:.4f}, acc={acc:.3f}")
    print(f"Random baseline: {1/vocab_size:.3f}")
    
    # Sample
    print("\nSample predictions:")
    preds = model.generate(tokens[:3])
    for i in range(3):
        print(f"  Input:  {tokens[i]}")
        print(f"  Target: {targets[i]}")
        print(f"  Pred:   {preds[i]}")
        print()
    
    # Save/load test
    print("Testing save/load...")
    model.save('/tmp/nat_mamba_test.npz')
    model2 = NATMamba.load('/tmp/nat_mamba_test.npz')
    _, acc2 = NATMambaTrainer(model2, lr=0.01).evaluate(tokens, targets)
    print(f"Loaded model accuracy: {acc2:.3f}")
    
    return model


if __name__ == '__main__':
    demo()
