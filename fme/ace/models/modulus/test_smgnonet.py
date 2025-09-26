#!/usr/bin/env python3
"""
Test script for SMgNO (Spherical Multigrid Neural Operator) implementation
"""

import torch
import torch.nn as nn
from smgnonet import SphericalMultigridNeuralOperatorNet


class MockParams:
    """Mock parameters class for testing"""
    def __init__(self):
        self.spectral_transform = "sht"
        self.img_shape_x = 64
        self.img_shape_y = 128
        self.scale_factor = 1
        self.N_in_channels = 3
        self.N_out_channels = 3
        self.embed_dim = 64
        self.num_layers = 2
        self.max_levels = 3
        self.smoothing_iterations = 2
        self.use_cshfs = True
        self.big_skip = True
        self.pos_embed = True
        self.encoder_layers = 1
        self.checkpointing = 0
        self.hard_thresholding_fraction = 1.0
        self.data_grid = "equiangular"


def test_smgno_basic():
    """Basic functionality test"""
    print("Testing SMgNO basic functionality...")
    
    # Create mock parameters
    params = MockParams()
    
    # Create model
    try:
        model = SphericalMultigridNeuralOperatorNet(
            params=params,
            img_shape=(64, 128),
            in_chans=3,
            out_chans=3,
            embed_dim=64,
            num_layers=2,
            max_levels=2,  # Reduced for testing
            smoothing_iterations=1,  # Reduced for testing
            use_cshfs=True
        )
        print("✓ Model creation successful")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    # Test forward pass
    try:
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 128)
        
        with torch.no_grad():
            output = model(x)
        
        expected_shape = (batch_size, 3, 64, 128)
        if output.shape == expected_shape:
            print(f"✓ Forward pass successful, output shape: {output.shape}")
        else:
            print(f"✗ Forward pass failed, expected shape: {expected_shape}, got: {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    return True


def test_smgno_components():
    """Test individual components"""
    print("\nTesting SMgNO components...")
    
    from smgnonet import (
        ConvolutionBasedSphericalHarmonicFunctions,
        MultigridSmoothingOperator,
        MultigridRestrictionOperator,
        MultigridProlongationOperator
    )
    
    # Mock transforms
    import torch_harmonics as th
    
    forward_transform = th.RealSHT(32, 64, lmax=16, mmax=32, grid="legendre-gauss").float()
    inverse_transform = th.InverseRealSHT(32, 64, lmax=16, mmax=32, grid="legendre-gauss").float()
    
    try:
        # Test CSHFs (Updated interface)
        cshf = ConvolutionBasedSphericalHarmonicFunctions(
            forward_transform, inverse_transform, 16, 16,
            use_spectral_conv=True, learnable_alpha=True
        )
        x = torch.randn(1, 16, 32, 64)
        output = cshf(x)
        print(f"✓ CSHFs test passed, output shape: {output.shape}")
        
        # Test Smoothing Operator (Updated interface - only takes previous solution)
        smoother = MultigridSmoothingOperator(
            forward_transform, inverse_transform, 16, smoothing_iterations=1
        )
        u = torch.randn(1, 16, 32, 64)
        smoothed = smoother(u)  # Only takes u_prev now
        print(f"✓ Smoothing operator test passed, output shape: {smoothed.shape}")
        
        # Test Restriction Operator
        restrictor = MultigridRestrictionOperator(16)
        restricted = restrictor(x)
        print(f"✓ Restriction operator test passed, output shape: {restricted.shape}")
        
        # Test Prolongation Operator
        prolongator = MultigridProlongationOperator(16)
        prolongated = prolongator(restricted)
        print(f"✓ Prolongation operator test passed, output shape: {prolongated.shape}")
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        return False
    
    return True


def test_smgno_gradients():
    """Test gradient computation"""
    print("\nTesting SMgNO gradient computation...")
    
    params = MockParams()
    params.embed_dim = 32  # Smaller for testing
    params.num_layers = 1
    params.max_levels = 2
    
    try:
        model = SphericalMultigridNeuralOperatorNet(
            params=params,
            img_shape=(32, 64),
            in_chans=2,
            out_chans=2,
            embed_dim=32,
            num_layers=1,
            max_levels=2,
            smoothing_iterations=1
        )
        
        x = torch.randn(1, 2, 32, 64, requires_grad=True)
        target = torch.randn(1, 2, 32, 64)
        
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check if gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters())
        if has_gradients:
            print("✓ Gradient computation successful")
        else:
            print("✗ No gradients computed")
            return False
            
    except Exception as e:
        print(f"✗ Gradient test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("SMgNO Implementation Test Suite")
    print("=" * 40)
    
    tests = [
        test_smgno_basic,
        test_smgno_components,
        test_smgno_gradients
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SMgNO implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
