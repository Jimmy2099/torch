package tensor

import (
	"reflect"
	"testing"
	// Assuming "fmt" might be needed if Tensor String() method exists for logging
	// "fmt"
)

// --- Assume Tensor struct and basic helpers exist ---
// type Tensor struct { Data []float32; Shape []int }
// func checkPanic(t *testing.T, f func()) { ... }

// --- Helper function needed by getBroadcastedShape ---
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// --- Tests ---

func Test_computeStrides(t *testing.T) {
	tests := []struct {
		name        string
		shape       []int
		wantStrides []int
	}{
		{"Scalar", []int{1}, []int{1}},
		{"Vector", []int{5}, []int{1}},
		{"Matrix 2x3", []int{2, 3}, []int{3, 1}},
		{"Matrix 3x2", []int{3, 2}, []int{2, 1}},
		{"3D Tensor", []int{2, 3, 4}, []int{12, 4, 1}},
		{"Shape with 1s", []int{1, 5, 1}, []int{5, 1, 1}},
		{"Leading 1", []int{1, 2, 3}, []int{6, 3, 1}},
		{"Trailing 1", []int{2, 3, 1}, []int{3, 1, 1}},
		{"Empty Shape", []int{}, []int{}},
		// Note: Implementation handles 0 dim by making subsequent strides 0,
		// which might be unexpected depending on desired stride definition.
		// Stride usually means "bytes/elements to jump to get to next element in this dim".
		// For dim 0 in [2,0,3], stride is 0*3 = 0?
		// The provided code calculates: stride[2]=1, stride=1*4=4? No shape[i] used
		// Let's re-read computeStrides:
		// strides[i] = stride // Stride needed to jump OVER dimension i+1 onwards
		// stride *= shape[i] // Update total size for next stride calc
		// Example: [2, 3, 4]
		// i=2: strides[2]=1, stride=1*4=4
		// i=1: strides[1]=4, stride=4*3=12
		// i=0: strides[0]=12, stride=12*2=24
		// Correct: [12, 4, 1]
		// Example: [2, 0, 3]
		// i=2: strides[2]=1, stride=1*3=3
		// i=1: strides[1]=3, stride=3*0=0
		// i=0: strides[0]=0, stride=0*2=0
		// Result: [0, 3, 1] - Testing this behavior.
		{"Shape with 0", []int{2, 0, 3}, []int{0, 3, 1}},
		{"Shape with 0 at end", []int{3, 2, 0}, []int{0, 0, 1}},
		{"Nil Shape", nil, []int{}}, // len(nil) is 0
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotStrides := computeStrides(tt.shape); !reflect.DeepEqual(gotStrides, tt.wantStrides) {
				t.Errorf("computeStrides(%v) = %v, want %v", tt.shape, gotStrides, tt.wantStrides)
			}
		})
	}
}

func Test_canBroadcast(t *testing.T) {
	tests := []struct {
		name string
		a    []int
		b    []int
		want bool
	}{
		{"Identical Shapes", []int{2, 3}, []int{2, 3}, true},
		{"Scalar and Matrix", []int{1}, []int{4, 5}, true},
		{"Matrix and Scalar", []int{4, 5}, []int{1}, true},
		{"Vector and Matrix (compatible)", []int{3}, []int{2, 3}, true},
		{"Matrix and Vector (compatible)", []int{2, 3}, []int{3}, true},
		{"Matrix and Vector (prefix compatible)", []int{2, 3}, []int{2, 1}, true}, // [2,3] vs [2,1] -> [2,3]
		{"Shapes need expansion", []int{5, 1, 4}, []int{1, 3, 1}, true},           // -> [5, 3, 4]
		{"Different Ranks, compatible", []int{4, 1}, []int{3, 4, 5}, false},       // 1 vs 5, 4 vs 4, _ vs 3
		{"Different Ranks, compatible 2", []int{4, 5}, []int{3, 1, 5}, true},      // 5 vs 5, 4 vs 1, _ vs 3 -> [3,4,5]
		{"Incompatible dimensions", []int{2, 3}, []int{2, 4}, false},
		{"Incompatible ranks and dimensions", []int{2, 3}, []int{4, 5, 6}, false},
		{"Empty and NonEmpty", []int{}, []int{2, 3}, true},
		{"NonEmpty and Empty", []int{2, 3}, []int{}, true},
		{"Both Empty", []int{}, []int{}, true},
		{"Nil and Empty", nil, []int{}, true}, // nil treated as len 0
		{"Nil and NonEmpty", nil, []int{2, 3}, true},
		{"Both Nil", nil, nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := canBroadcast(tt.a, tt.b); got != tt.want {
				t.Errorf("canBroadcast(%v, %v) = %v, want %v", tt.a, tt.b, got, tt.want)
			}
			// Test symmetry
			if got := canBroadcast(tt.b, tt.a); got != tt.want {
				t.Errorf("canBroadcast(%v, %v) [Symmetry check] = %v, want %v", tt.b, tt.a, got, tt.want)
			}
		})
	}
}

func Test_getBroadcastedShape(t *testing.T) {
	tests := []struct {
		name      string
		a         []int
		b         []int
		wantShape []int
		wantErr   bool // Expect panic
	}{
		{"Identical Shapes", []int{2, 3}, []int{2, 3}, []int{2, 3}, false},
		{"Scalar and Matrix", []int{1}, []int{4, 5}, []int{4, 5}, false},
		{"Matrix and Scalar", []int{4, 5}, []int{1}, []int{4, 5}, false},
		{"Vector and Matrix (compatible)", []int{3}, []int{2, 3}, []int{2, 3}, false},
		{"Matrix and Vector (compatible)", []int{2, 3}, []int{3}, []int{2, 3}, false},
		{"Matrix and Vector (prefix compatible)", []int{2, 3}, []int{2, 1}, []int{2, 3}, false},
		{"Shapes need expansion", []int{5, 1, 4}, []int{1, 3, 1}, []int{5, 3, 4}, false},
		{"Different Ranks, compatible", []int{4, 5}, []int{3, 1, 5}, []int{3, 4, 5}, false},
		{"Empty and NonEmpty", []int{}, []int{2, 3}, []int{2, 3}, false},
		{"NonEmpty and Empty", []int{2, 3}, []int{}, []int{2, 3}, false},
		{"Both Empty", []int{}, []int{}, []int{}, false},
		{"Nil and Empty", nil, []int{}, []int{}, false},
		{"Nil and NonEmpty", nil, []int{2, 3}, []int{2, 3}, false},
		{"Both Nil", nil, nil, []int{}, false}, // Result is empty shape

		{"Incompatible dimensions", []int{2, 3}, []int{2, 4}, nil, true},
		{"Incompatible ranks and dimensions", []int{2, 3}, []int{4, 5, 6}, nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.wantErr {
				checkPanic(t, func() { getBroadcastedShape(tt.a, tt.b) }, "无法广播形状")
			} else {
				gotShape := getBroadcastedShape(tt.a, tt.b)
				if !reflect.DeepEqual(gotShape, tt.wantShape) {
					t.Errorf("getBroadcastedShape(%v, %v) = %v, want %v", tt.a, tt.b, gotShape, tt.wantShape)
				}
				// Test symmetry if not error case
				gotShapeSym := getBroadcastedShape(tt.b, tt.a)
				if !reflect.DeepEqual(gotShapeSym, tt.wantShape) {
					t.Errorf("getBroadcastedShape(%v, %v) [Symmetry check] = %v, want %v", tt.b, tt.a, gotShapeSym, tt.wantShape)
				}
			}
		})
	}
}

func TestTensor_broadcastedIndex(t *testing.T) {
	// Note: This function's behavior assumes len(indices) == len(t.Shape) == len(strides).
	// It calculates the linear index in the *source* tensor `t` corresponding to the
	// multi-dimensional `indices` (from the broadcasted result), skipping dimensions
	// where t.Shape[i] == 1.

	tests := []struct {
		name          string
		sourceShape   []int
		targetIndices []int // Indices in the (conceptual) broadcasted target
		wantIndex     int   // Expected linear index in the source tensor's data
	}{
		// Source [1, 3], Target Shape (e.g., [N, 3]), Strides [3, 1]
		{"Source Dim 1 is 1", []int{1, 3}, []int{0, 0}, 0},
		{"Source Dim 1 is 1, Idx 1", []int{1, 3}, []int{0, 1}, 1},
		{"Source Dim 1 is 1, Idx 2", []int{1, 3}, []int{0, 2}, 2},
		{"Source Dim 1 is 1, Target Idx 0 changed", []int{1, 3}, []int{5, 1}, 1}, // Target index[0] ignored as source dim is 1

		// Source [3, 1], Target Shape (e.g., [3, M]), Strides [1, 1]
		{"Source Dim 2 is 1", []int{3, 1}, []int{0, 0}, 0},
		{"Source Dim 2 is 1, Idx 0", []int{3, 1}, []int{1, 0}, 1},
		{"Source Dim 2 is 1, Idx 1", []int{3, 1}, []int{2, 0}, 2},
		{"Source Dim 2 is 1, Target Idx 1 changed", []int{3, 1}, []int{1, 5}, 1}, // Target index[1] ignored as source dim is 1

		// Source [1, 1], Target Shape (e.g., [N, M]), Strides [1, 1]
		{"Source All Dims 1", []int{1, 1}, []int{0, 0}, 0},
		{"Source All Dims 1, Indices Vary", []int{1, 1}, []int{5, 8}, 0}, // Always maps to index 0

		// Source [2, 3], Target Shape [2, 3], Strides [3, 1] (No broadcasting needed)
		{"No Broadcasting Needed", []int{2, 3}, []int{0, 0}, 0},
		{"No Broadcasting Needed 1", []int{2, 3}, []int{0, 2}, 2},
		{"No Broadcasting Needed 2", []int{2, 3}, []int{1, 0}, 3},
		{"No Broadcasting Needed 3", []int{2, 3}, []int{1, 2}, 5},

		// Source [1, 2, 1, 3], Target Shape (e.g., [N, 2, M, 3]), Strides [6, 3, 3, 1]
		{"Mixed 1s and >1s", []int{1, 2, 1, 3}, []int{0, 0, 0, 0}, 0},       // Skips dim 0, 2. Adds 0*3(dim1) + 0*1(dim3) = 0
		{"Mixed 1s and >1s Idx 1", []int{1, 2, 1, 3}, []int{5, 1, 0, 0}, 3}, // Skips 0, 2. Adds 1*3(dim1) + 0*1(dim3) = 3
		{"Mixed 1s and >1s Idx 2", []int{1, 2, 1, 3}, []int{5, 0, 9, 1}, 1}, // Skips 0, 2. Adds 0*3(dim1) + 1*1(dim3) = 1
		{"Mixed 1s and >1s Idx 3", []int{1, 2, 1, 3}, []int{5, 1, 9, 2}, 5}, // Skips 0, 2. Adds 1*3(dim1) + 2*1(dim3) = 3+2 = 5
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a dummy tensor with the source shape (data not used, but need shape)
			// Size calculation needed only for Data allocation if we were using NewTensor
			tensor := &Tensor{Shape: tt.sourceShape}
			// Calculate strides corresponding to the source shape
			strides := computeStrides(tt.sourceShape)

			gotIndex := tensor.broadcastedIndex(tt.targetIndices, strides)

			if gotIndex != tt.wantIndex {
				t.Errorf("broadcastedIndex(%v, %v) with source shape %v = %d, want %d", tt.targetIndices, strides, tt.sourceShape, gotIndex, tt.wantIndex)
			}
		})
	}
	// Test panic case if lengths don't match (though function assumes they do)
	t.Run("PanicMismatchedLengths", func(t *testing.T) {
		tensor := &Tensor{Shape: []int{2, 3}}
		strides := []int{3, 1}
		badIndices := []int{1, 1, 1} // Length 3 != Length 2
		// Expect panic due to index out of range on t.Shape[i] or strides[i]
		checkPanic(t, func() { tensor.broadcastedIndex(badIndices, strides) }, "")

		badStrides := []int{1} // Length 1 != Length 2
		goodIndices := []int{1, 1}
		checkPanic(t, func() { tensor.broadcastedIndex(goodIndices, badStrides) }, "")
	})

}
