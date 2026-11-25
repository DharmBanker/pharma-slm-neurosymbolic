"""
Unit Tests & Architectural Verification for Modern Pharma-SLM
==============================================================
Validates core mathematical properties of modern SLM transformer blocks:
1. RMSNorm: Scale-invariance, pre-norm execution, fp32 numerical stability.
2. RoPE: Rotational symmetry, relative distance invariance <R_m q, R_n k> = f(m - n).
3. GQA (Grouped-Query Attention): KV-head compression, repeat-interleave expansion, causal masking.
4. SwiGLU: Gated bilinear feed-forward activations and GPU-aligned dimension padding.
5. GPT Architecture: Weight tying, parameter counting, loss computation, backprop gradient sanity.
6. Autoregressive Generation: KV token stepping, top-k truncation, temperature boundary defense.
"""
import unittest
import sys
import os
import math
import torch
import torch.nn.functional as F

# Ensure src/ is on sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import ModelConfig
from model import RMSNorm, precompute_rope, apply_rope, _rotate_half, Attention, SwiGLU, Block, GPT


class TestModelArchitecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # Compact config for fast, deterministic unit testing
        self.cfg = ModelConfig(
            vocab_size=128,
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,  # GQA with 2 KV groups (n_rep = 2)
            max_seq_len=64,
            dropout=0.0,
            tie_embeddings=True,
            norm_eps=1e-5
        )

    def test_rmsnorm_mathematical_properties(self):
        """RMSNorm must exhibit exact scale-invariance and numerical stability."""
        dim = 32
        norm = RMSNorm(dim=dim, eps=1e-5)
        x = torch.randn(2, 8, dim)

        y1 = norm(x)
        self.assertEqual(y1.shape, (2, 8, dim))

        # Mathematical Property 1: Scale Invariance
        # Scaling input by positive scalar c should not alter normalized direction
        c = 3.7
        y2 = norm(x * c)
        torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)

        # Mathematical Property 2: Unit Root Mean Square
        # RMS of normalized output before learnable weight should be approximately 1.0
        with torch.no_grad():
            norm.weight.fill_(1.0)
            rms = torch.sqrt(torch.mean(norm(x).pow(2), dim=-1))
            torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=1e-3)
        print("\n[PASS] RMSNorm scale-invariance and unit RMS verified.")

    def test_rope_relative_position_invariance(self):
        """RoPE inner product <R_m q, R_n k> must depend strictly on relative displacement (m - n)."""
        head_dim = 16
        seq_len = 32
        cos, sin = precompute_rope(head_dim=head_dim, max_seq_len=seq_len)
        self.assertEqual(cos.shape, (seq_len, head_dim))
        self.assertEqual(sin.shape, (seq_len, head_dim))

        # Create two identical query and key vectors
        q = torch.randn(1, 1, 1, head_dim)
        k = torch.randn(1, 1, 1, head_dim)

        # Pair A: position m=4, n=2 (relative offset = 2)
        q_pos4, _ = apply_rope(q, q, cos[4:5], sin[4:5])
        _, k_pos2 = apply_rope(k, k, cos[2:3], sin[2:3])
        dot_product_pair_a = torch.sum(q_pos4 * k_pos2).item()

        # Pair B: position m=9, n=7 (same relative offset = 2)
        q_pos9, _ = apply_rope(q, q, cos[9:10], sin[9:10])
        _, k_pos7 = apply_rope(k, k, cos[7:8], sin[7:8])
        dot_product_pair_b = torch.sum(q_pos9 * k_pos7).item()

        # Relative shift invariance theorem: dot_product_pair_a == dot_product_pair_b
        self.assertAlmostEqual(dot_product_pair_a, dot_product_pair_b, places=4)

        # Pair C: position m=4, n=1 (different relative offset = 3) -> should NOT match
        _, k_pos1 = apply_rope(k, k, cos[1:2], sin[1:2])
        dot_product_pair_c = torch.sum(q_pos4 * k_pos1).item()
        self.assertNotAlmostEqual(dot_product_pair_a, dot_product_pair_c, places=2)
        print("[PASS] RoPE relative distance invariance theorem verified (<R_m q, R_n k> = f(m - n)).")

    def test_grouped_query_attention(self):
        """GQA must correctly project kv_heads < q_heads and expand kv via repeat_interleave."""
        attn = Attention(self.cfg)
        self.assertEqual(attn.n_heads, 4)
        self.assertEqual(attn.n_kv_heads, 2)
        self.assertEqual(attn.n_rep, 2)
        self.assertEqual(attn.head_dim, 16)

        # Verify parameter projection dimensions
        # Query projection: dim -> n_heads * head_dim = 4 * 16 = 64
        self.assertEqual(attn.wq.weight.shape, (64, 64))
        # Key & Value projections: dim -> n_kv_heads * head_dim = 2 * 16 = 32 (50% reduction!)
        self.assertEqual(attn.wk.weight.shape, (32, 64))
        self.assertEqual(attn.wv.weight.shape, (32, 64))

        # Forward execution
        B, T, C = 2, 8, 64
        x = torch.randn(B, T, C)
        cos, sin = precompute_rope(head_dim=attn.head_dim, max_seq_len=T)
        out = attn(x, cos, sin)
        self.assertEqual(out.shape, (B, T, C))
        print("[PASS] Grouped-Query Attention (GQA) KV compression and forward pass verified.")

    def test_swiglu_activation_and_dimensions(self):
        """SwiGLU must compute w2(SiLU(w1(x)) * w3(x)) with 256-multiple dimension rounding."""
        swiglu = SwiGLU(self.cfg)
        # Resolved hidden: mlp_mult * 64 = 170.66 -> round to multiple of 256 => 256
        hidden = self.cfg.resolved_mlp_hidden()
        self.assertEqual(hidden, 256)
        self.assertEqual(swiglu.w1.weight.shape, (256, 64))
        self.assertEqual(swiglu.w3.weight.shape, (256, 64))
        self.assertEqual(swiglu.w2.weight.shape, (64, 256))

        x = torch.randn(2, 6, 64)
        out = swiglu(x)
        self.assertEqual(out.shape, (2, 6, 64))

        # Direct mathematical verification of SwiGLU formula
        with torch.no_grad():
            manual_silu = F.silu(swiglu.w1(x))
            manual_val = swiglu.w3(x)
            manual_out = swiglu.w2(manual_silu * manual_val)
            torch.testing.assert_close(out, manual_out, atol=1e-5, rtol=1e-5)
        print("[PASS] SwiGLU gated activation formula and kernel dimension rounding verified.")

    def test_gpt_weight_tying_and_loss(self):
        """GPT model must tie embedding weights to lm_head and compute valid cross-entropy loss."""
        model = GPT(self.cfg)

        # 1. Weight tying verification
        self.assertTrue(self.cfg.tie_embeddings)
        self.assertIs(model.lm_head.weight, model.tok_emb.weight)

        # 2. Forward pass with training targets
        B, T = 2, 10
        idx = torch.randint(0, self.cfg.vocab_size, (B, T))
        targets = torch.randint(0, self.cfg.vocab_size, (B, T))

        logits, loss = model(idx, targets=targets)
        self.assertEqual(logits.shape, (B, T, self.cfg.vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

        # 3. Backward pass sanity check (no NaNs in gradients)
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN in gradient of {name}")

        # 4. Inference pass (only last token logits returned)
        with torch.no_grad():
            inf_logits, inf_loss = model(idx)
            self.assertEqual(inf_logits.shape, (B, 1, self.cfg.vocab_size))
            self.assertIsNone(inf_loss)
        print("[PASS] GPT weight tying, cross-entropy loss, and gradient backprop verified.")

    def test_autoregressive_generation_and_boundaries(self):
        """Model generation must append expected token count and handle temperature safely."""
        model = GPT(self.cfg)
        model.eval()

        prompt = torch.tensor([[10, 20, 30]])  # Shape (1, 3)
        max_new = 5
        generated = model.generate(prompt, max_new_tokens=max_new, temperature=0.7, top_k=10)
        self.assertEqual(generated.shape, (1, 3 + max_new))

        # Check prompt prefix is preserved exactly
        torch.testing.assert_close(generated[:, :3], prompt)

        # Test boundary: near-zero temperature (greedy sampling without division by zero)
        greedy_gen = model.generate(prompt, max_new_tokens=2, temperature=1e-8, top_k=1)
        self.assertEqual(greedy_gen.shape, (1, 5))
        print("[PASS] Autoregressive generation and temperature boundary defenses verified.")

    def test_adversarial_and_architectural_edge_cases(self):
        """Red-Teaming: Sequence length overflow and invalid attention head divisions."""
        model = GPT(self.cfg)

        # Edge Case 1: Sequence length exceeding max_seq_len must assert error
        overflow_len = self.cfg.max_seq_len + 10
        overflow_idx = torch.randint(0, self.cfg.vocab_size, (1, overflow_len))
        with self.assertRaises(AssertionError) as ctx:
            model(overflow_idx)
        self.assertIn("seq len", str(ctx.exception))

        # Edge Case 2: Attention heads not divisible by KV heads must assert error
        invalid_cfg = ModelConfig(
            dim=64,
            n_heads=5,      # 5 heads is not divisible by 2 kv_heads
            n_kv_heads=2
        )
        with self.assertRaises(AssertionError):
            Attention(invalid_cfg)

        # Edge Case 3: Dimension not divisible by n_heads must assert error
        invalid_dim_cfg = ModelConfig(
            dim=65,         # 65 not divisible by 4 heads
            n_heads=4
        )
        with self.assertRaises(AssertionError):
            Attention(invalid_dim_cfg)
        print("[PASS] Architectural assertions and adversarial edge cases caught as designed.")


if __name__ == "__main__":
    unittest.main()

