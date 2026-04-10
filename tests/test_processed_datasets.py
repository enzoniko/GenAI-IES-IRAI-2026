import unittest
import torch
import os
import sys

# Add project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestProcessedDatasets(unittest.TestCase):
    def test_normal_v1_datasets(self):
        # Base directory is one level up from tests
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Correct path for v1 processed MaFaulDa
        v1_dir = os.path.join(base_dir, "data", "processed-mafaulda", "v1")
        
        x_path = os.path.join(v1_dir, "X_normal_v1_trainingset.pth")
        y_path = os.path.join(v1_dir, "Y_normal_v1_trainingset.pth")
        
        # Verify file existence
        self.assertTrue(os.path.exists(x_path), f"File missing: {x_path}")
        self.assertTrue(os.path.exists(y_path), f"File missing: {y_path}")
        
        # Load datasets onto CPU to avoid VRAM overhead in testing
        X = torch.load(x_path, map_location='cpu')
        Y = torch.load(y_path, map_location='cpu')
        
        # Verify variable types
        self.assertIsInstance(X, torch.Tensor, "X is not a torch tensor")
        self.assertIsInstance(Y, torch.Tensor, "Y is not a torch tensor")
        
        # Verify basic shapes are 3D
        self.assertEqual(X.dim(), 3, "X should be a 3D tensor [samples, window_size, features]")
        self.assertEqual(Y.dim(), 3, "Y should be a 3D tensor [samples, window_size, features]")
        
        # Verify correct number of features
        # 10 features from clean_mafaulda_processor.py: Velocities (4), Positions (4), Omega (1), Time (1)
        self.assertEqual(X.shape[-1], 10, "X should strictly have 10 features") 
        # 4 outputs: Accelerations (4)
        self.assertEqual(Y.shape[-1], 4, "Y should strictly have 4 features") 
        
        # Verify samples and window size matching between X and Y
        self.assertEqual(X.shape[0], Y.shape[0], "Batch/Sample dimension mismatch between X and Y")
        self.assertEqual(X.shape[1], Y.shape[1], "Sequence length mismatch between X and Y")

if __name__ == '__main__':
    unittest.main()
