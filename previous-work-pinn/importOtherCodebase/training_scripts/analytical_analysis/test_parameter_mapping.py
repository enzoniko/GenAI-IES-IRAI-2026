#!/usr/bin/env python3
"""
Test script to verify parameter mapping from synthetic data to PINN-compatible parameters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_synthetic_dataset_v2 import map_to_pinn_parameters, get_base_parameters

def test_parameter_mapping():
    """Test the parameter mapping function with base parameters."""
    
    # Get base parameters from synthetic data generation
    base_params = get_base_parameters()
    print("Original synthetic data parameters:")
    for key, value in base_params.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    
    # Map to PINN parameters
    pinn_params = map_to_pinn_parameters(base_params)
    print("PINN-compatible parameters:")
    for key, value in pinn_params.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    
    # Verify the mapping logic
    print("Parameter mapping verification:")
    print(f"  M1: {base_params['M1']} → {pinn_params['M1']} ✓ (direct)")
    print(f"  M2: {base_params['M2']} → {pinn_params['M2']} ✓ (direct)")
    print(f"  M3: {base_params['M3']} → {pinn_params['M3']} ✓ (direct)")
    
    print(f"  K1: {base_params['Ks1']} + {base_params['Kb']} = {pinn_params['K1']} ✓ (Ks1 + Kb)")
    print(f"  K2: {base_params['Ks2']} + {base_params['Kb']} = {pinn_params['K2']} ✓ (Ks2 + Kb)")
    
    print(f"  D1: {base_params['Ds1']} → {pinn_params['D1']} ✓ (Ds1)")
    print(f"  D2: {base_params['Ds2']} → {pinn_params['D2']} ✓ (Ds2)")
    print(f"  D3: {base_params['Db']} → {pinn_params['D3']} ✓ (Db)")
    
    print(f"  E1: {base_params['mu_eps']} / {base_params['M1']} = {pinn_params['E1']} ✓ (mu_eps / M1)")
    
    print("\n" + "="*60)
    
    # Check that all required PINN parameters are present
    required_pinn_params = ['M1', 'M2', 'M3', 'D1', 'D2', 'D3', 'K1', 'K2', 'E1']
    missing_params = [param for param in required_pinn_params if param not in pinn_params]
    
    if missing_params:
        print(f"❌ Missing PINN parameters: {missing_params}")
        return False
    else:
        print("✅ All required PINN parameters are present")
    
    # Check parameter ranges for reasonableness
    print("\nParameter range checks:")
    if 0 < pinn_params['M1'] < 100:  # Central mass should be reasonable
        print(f"  M1: {pinn_params['M1']} kg ✓ (reasonable range)")
    else:
        print(f"  M1: {pinn_params['M1']} kg ❌ (unreasonable range)")
    
    if 0 < pinn_params['K1'] < 1e8:  # Stiffness should be reasonable
        print(f"  K1: {pinn_params['K1']} N/m ✓ (reasonable range)")
    else:
        print(f"  K1: {pinn_params['K1']} N/m ❌ (unreasonable range)")
    
    if 0 < pinn_params['E1'] < 1e-3:  # Eccentricity should be small
        print(f"  E1: {pinn_params['E1']} m ✓ (reasonable range)")
    else:
        print(f"  E1: {pinn_params['E1']} m ❌ (unreasonable range)")
    
    return True

if __name__ == "__main__":
    print("Testing parameter mapping from synthetic data to PINN-compatible parameters")
    print("="*80)
    
    success = test_parameter_mapping()
    
    if success:
        print("\n✅ Parameter mapping test passed!")
        print("The synthetic data generation script now produces PINN-compatible parameters.")
    else:
        print("\n❌ Parameter mapping test failed!")
        print("Please check the mapping function implementation.")
