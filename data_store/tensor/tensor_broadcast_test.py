import numpy as np

def print_test_result(shape_a, shape_b, result):
    """Helper function to print test results"""
    print(f"\n=== Test: {shape_a} + {shape_b} ===")
    print(f"Broadcasted shape: {result.shape}")
    print("Sample values (first 5 elements):")
    print(result.ravel()[:5])
    print("Full output shape:", result.shape)

def run_broadcast_test(shape_a, shape_b):
    """Run a single broadcast test and print results"""
    try:
        print(f"\n{'='*50}")
        print(f"TESTING BROADCAST: {shape_a} + {shape_b}")
        print('='*50)

        a = np.ones(shape_a)
        b = np.ones(shape_b)
        result = a + b

        print("\n[SUCCESS] Shapes are compatible")
        print_test_result(shape_a, shape_b, result)
        return True

    except ValueError as e:
        print("\n[FAILED] Shapes are incompatible")
        print(f"Error: {str(e)}")
        return False

# Test cases
test_cases = [
    # Same shape
    ((2, 3), (2, 3)),

    # Trailing dimension broadcast
    ((4, 3), (3,)),

    # Middle dimension broadcast
    ((2, 1, 4), (2, 3, 4)),

    # High-dim broadcast (original issue)
    ((1, 256, 1, 1), (256,)),

    # Scalar broadcast
    ((2, 2), (1,)),

    # Empty tensor
    ((0,), (1,)),
    ((1,), (0,)),
    # Incompatible shapes (should fail)
    ((3, 4), (2, 3)),

    # 3D broadcast
    ((3, 1, 5), (1, 4, 1)),

    # 4D broadcast
    ((1, 16, 1, 8), (16, 1, 8))
]

if __name__ == "__main__":
    print("=== Tensor Broadcast Test Suite ===")
    print(f"NumPy version: {np.__version__}")
    print(f"Testing {len(test_cases)} cases...\n")

    for i, (shape_a, shape_b) in enumerate(test_cases, 1):
        print(f"\n{'#'*20} Test Case {i} {'#'*20}")
        success = run_broadcast_test(shape_a, shape_b)

        if success:
            print("Result: PASSED")
        else:
            print("Result: FAILED (expected failure for incompatible shapes)")

    print("\n=== Test Suite Complete ===")
