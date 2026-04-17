# pyright: reportAny=false, reportPrivateImportUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

import torch
from src.models.oracles import PriorWorkOracle

def test_oracle_differentiability():
    print("--- Initializing PriorWorkOracle ---")
    oracle = PriorWorkOracle()
    oracle.eval()  # The Oracle must ALWAYS be in eval mode during SDEdit
    
    # 1. Create a mock generated vibration signal (Batch=2, Channels=4, Length=1639)
    batch_size = 2
    seq_len = 1639
    x_generated = torch.randn(batch_size, 4, seq_len, requires_grad=True)
    
    print(f"\n[Forward Pass]")
    print(f"Input shape: {x_generated.shape} (Fake Accelerations)")
    
    # Pass it through the physics engine
    y_emb = oracle(x_generated, omega=1200)
    
    print(f"Output shape: {y_emb.shape} (Physics Embeddings)")
    assert y_emb.shape == (batch_size, 2240), f"Expected shape {(batch_size, 2240)}, got {y_emb.shape}"
    print("Forward pass successful and mathematical shapes match!")
    
    print(f"\n[Backward Pass - Differentiability Test]")
    # Create a mock target (e.g., the embedding of a real Outer-Race fault)
    target_emb = torch.randn(batch_size, 2240)
    
    # Calculate the penalty (How far is the fake signal from the true physics?)
    penalty = torch.nn.functional.mse_loss(y_emb, target_emb)
    
    # Attempt to calculate the gradients backward through the PINN, 
    # through the double integration, and into the raw acceleration signal.
    grad = torch.autograd.grad(penalty, x_generated)[0]
    
    print(f"Gradient shape: {grad.shape}")
    assert grad.shape == x_generated.shape, "Gradient shape does not match input shape!"
    print("Backward pass successful! The entire physics pipeline is 100% differentiable.")

def test_oracle_target_buffers():
    print("\n--- Testing Target Distribution Buffers ---")
    oracle = PriorWorkOracle()
    dim = oracle.embed_dim
    
    # Test setting and getting targets for various classes
    test_classes = [0, 1, 3]
    for c in test_classes:
        mock_emb = torch.randn(dim)
        oracle.set_target_distribution(c, mock_emb)
        retrieved = oracle.get_target_distribution(c)
        assert torch.allclose(mock_emb, retrieved), f"Buffer mismatch for class {c}"
        print(f"Class {c} buffer verified.")
        
    # Test invalid index
    try:
        oracle.get_target_distribution(4)
        assert False, "Should have raised ValueError for index 4"
    except ValueError as e:
        print(f"Caught expected error for index 4: {e}")

if __name__ == '__main__':
    test_oracle_differentiability()
    test_oracle_target_buffers()
