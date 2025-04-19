package tensor

import (
	"reflect"
	"testing"
)

func TestNewTensor2(t *testing.T) {
	testCases := []struct {
		name           string
		inputData      []float32
		inputShape     []int
		expectedTensor *Tensor
		expectPanic    bool
	}{
		{
			name:       "Simple 1D Vector",
			inputData:  []float32{1.0, 2.0, 3.0},
			inputShape: []int{3},
			expectedTensor: &Tensor{
				Data:  []float32{1.0, 2.0, 3.0},
				shape: []int{3},
			},
			expectPanic: false,
		},
		{
			name:       "Simple 2D Matrix",
			inputData:  []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
			inputShape: []int{2, 3},
			expectedTensor: &Tensor{
				Data:  []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
				shape: []int{2, 3},
			},
			expectPanic: false,
		},
		{
			name:       "Empty Data and shape",
			inputData:  []float32{},
			inputShape: []int{},
			expectedTensor: &Tensor{
				Data:  []float32{},
				shape: []int{},
			},
			expectPanic: false,
		},
		{
			name:       "Nil Data and shape",
			inputData:  nil,
			inputShape: nil,
			expectedTensor: &Tensor{
				Data:  nil,
				shape: nil,
			},
			expectPanic: false,
		},
		{
			name:       "shape with Zero Dimension",
			inputData:  []float32{},
			inputShape: []int{2, 0, 3},
			expectedTensor: &Tensor{
				Data:  []float32{},
				shape: []int{2, 0, 3},
			},
			expectPanic: false,
		},
		{
			name:       "High Dimensional Tensor (3D)",
			inputData:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
			inputShape: []int{2, 2, 2},
			expectedTensor: &Tensor{
				Data:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				shape: []int{2, 2, 2},
			},
			expectPanic: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {

			defer func() {
				r := recover()
				if r != nil && !tc.expectPanic {
					t.Errorf("NewTensor panicked unexpectedly: %v", r)
				} else if r == nil && tc.expectPanic {
					t.Errorf("NewTensor was expected to panic but did not")
				}
			}()

			actualTensor := NewTensor(tc.inputData, tc.inputShape)

			if actualTensor == nil {
				t.Fatalf("NewTensor returned nil, expected a valid *Tensor")
			}

			if !reflect.DeepEqual(actualTensor.Data, tc.expectedTensor.Data) {
				isActualDataNilOrEmpty := actualTensor.Data == nil || len(actualTensor.Data) == 0
				isExpectedDataNilOrEmpty := tc.expectedTensor.Data == nil || len(tc.expectedTensor.Data) == 0
				if !(isActualDataNilOrEmpty && isExpectedDataNilOrEmpty) {
					t.Errorf("Data mismatch: expected %v, got %v", tc.expectedTensor.Data, actualTensor.Data)
				}
			}

			if !reflect.DeepEqual(actualTensor.shape, tc.expectedTensor.shape) {
				isActualShapeNilOrEmpty := actualTensor.shape == nil || len(actualTensor.shape) == 0
				isExpectedShapeNilOrEmpty := tc.expectedTensor.shape == nil || len(tc.expectedTensor.shape) == 0
				if !(isActualShapeNilOrEmpty && isExpectedShapeNilOrEmpty) {
					t.Errorf("shape mismatch: expected %v, got %v", tc.expectedTensor.shape, actualTensor.shape)
				}
			}

		})
	}
}
