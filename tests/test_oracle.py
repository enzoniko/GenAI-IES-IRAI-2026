import torch
from src.models.oracles import PriorWorkOracle

def test_oracle_differentiability():
    print("--- Initializing PriorWorkOracle ---")
    oracle = PriorWorkOracle()
    oracle.eval()  # The Oracle must ALWAYS be in eval mode during SDEdit
    
    # 1. Create a mock generated vibration signal (Batch=2, Channels=4, Length=5000)
    batch_size = 2
    x_generated = torch.randn(batch_size, 4, 5000, requires_grad=True)
    
    print(f"\n[Forward Pass]")
    print(f"Input shape: {x_generated.shape} (Fake Accelerations)")
    
    # Pass it through the physics engine
    y_emb = oracle(x_generated)
    
    print(f"Output shape: {y_emb.shape} (Physics Embeddings)")
    assert y_emb.shape == (batch_size, 64), f"Expected shape {(batch_size, 64)}, got {y_emb.shape}"
    print("Forward pass successful and mathematical shapes match!")
    
    print(f"\n[Backward Pass - Differentiability Test]")
    # Create a mock target (e.g., the embedding of a real Outer-Race fault)
    target_emb = torch.randn(batch_size, 64)
    
    # Calculate the penalty (How far is the fake signal from the true physics?)
    penalty = torch.nn.functional.mse_loss(y_emb, target_emb)
    
    # Attempt to calculate the gradients backward through the PINN, 
    # through the double integration, and into the raw acceleration signal.
    grad = torch.autograd.grad(penalty, x_generated)[0]
    
    print(f"Gradient shape: {grad.shape}")
    assert grad.shape == x_generated.shape, "Gradient shape does not match input shape!"
    print("Backward pass successful! The entire physics pipeline is 100% differentiable.")

if __name__ == '__main__':
    test_oracle_differentiability()
