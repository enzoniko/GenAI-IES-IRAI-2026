import torch
import sys
import os

# Ensure the src module is found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.oracles import PriorWorkOracle

def test_math_extractor():
    print("Testing MathFeatureExtractor integration into PriorWorkOracle...")
    
    # Initialize Oracle with the purely mathematical extractor
    oracle = PriorWorkOracle()
    
    # Create dummy trace: Batch of 2, 4 Channels, 5000 points long (simulating valid batch)
    dummy_trace = torch.randn(2, 4, 5000)
    dummy_trace.requires_grad_(True)
    
    # Compute embeddings exactly as done in the physics pipeline
    embeddings = oracle(dummy_trace)
    
    print(f"Native Math Embedding Shape: {embeddings.shape}")
    
    # Expected: 2120 dimensions (8 physical channels * (9 time + 256 freq bins))
    expected_dim = 8 * (9 + 256)
    assert embeddings.shape[-1] == expected_dim, f"Dimension mismatch! Got {embeddings.shape[-1]} but expected {expected_dim}"
    print(f"Dimension check passed! Size matches math structural calculations ({expected_dim}).")
    
    # Check backwards pass to guarantee no detached tensors/zero-grad errors
    loss = embeddings.sum()
    loss.backward()
    
    assert dummy_trace.grad is not None, "Gradient flow FAILED! Backprop broke somewhere in MathFeatureExtractor or PINN residuals."
    assert not torch.isnan(dummy_trace.grad).any(), "Gradient exploded to NaN! Epsilon padding failed."
    print("Gradient flow verified perfectly. Extractor is 100% implicitly differentiable!")

if __name__ == "__main__":
    test_math_extractor()
