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
				Shape: []int{3},
			},
			expectPanic: false,
		},
		{
			name:       "Simple 2D Matrix",
			inputData:  []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
			inputShape: []int{2, 3},
			expectedTensor: &Tensor{
				Data:  []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
				Shape: []int{2, 3},
			},
			expectPanic: false,
		},
		{
			name:       "Empty Data and Shape",
			inputData:  []float32{},
			inputShape: []int{},
			expectedTensor: &Tensor{
				Data:  []float32{},
				Shape: []int{},
			},
			expectPanic: false,
		},
		{
			name:       "Nil Data and Shape",
			inputData:  nil,
			inputShape: nil,
			expectedTensor: &Tensor{
				Data:  nil,
				Shape: nil,
			},
			expectPanic: false,
		},
		{
			name:       "Shape with Zero Dimension",
			inputData:  []float32{},
			inputShape: []int{2, 0, 3},
			expectedTensor: &Tensor{
				Data:  []float32{},
				Shape: []int{2, 0, 3},
			},
			expectPanic: false,
		},
		{
			name:       "High Dimensional Tensor (3D)",
			inputData:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
			inputShape: []int{2, 2, 2},
			expectedTensor: &Tensor{
				Data:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
				Shape: []int{2, 2, 2},
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

			if !reflect.DeepEqual(actualTensor.Shape, tc.expectedTensor.Shape) {
				isActualShapeNilOrEmpty := actualTensor.Shape == nil || len(actualTensor.Shape) == 0
				isExpectedShapeNilOrEmpty := tc.expectedTensor.Shape == nil || len(tc.expectedTensor.Shape) == 0
				if !(isActualShapeNilOrEmpty && isExpectedShapeNilOrEmpty) {
					t.Errorf("Shape mismatch: expected %v, got %v", tc.expectedTensor.Shape, actualTensor.Shape)
				}
			}


		})
	}
}
