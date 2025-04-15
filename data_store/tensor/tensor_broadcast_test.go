package tensor

import (
	"reflect"
	"testing"
)


func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}


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
		{"Shape with 0", []int{2, 0, 3}, []int{0, 3, 1}},
		{"Shape with 0 at end", []int{3, 2, 0}, []int{0, 0, 1}},
		{"Nil Shape", nil, []int{}},
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
		{"Matrix and Vector (prefix compatible)", []int{2, 3}, []int{2, 1}, true},
		{"Shapes need expansion", []int{5, 1, 4}, []int{1, 3, 1}, true},
		{"Different Ranks, compatible", []int{4, 1}, []int{3, 4, 5}, false},
		{"Different Ranks, compatible 2", []int{4, 5}, []int{3, 1, 5}, true},
		{"Incompatible dimensions", []int{2, 3}, []int{2, 4}, false},
		{"Incompatible ranks and dimensions", []int{2, 3}, []int{4, 5, 6}, false},
		{"Empty and NonEmpty", []int{}, []int{2, 3}, true},
		{"NonEmpty and Empty", []int{2, 3}, []int{}, true},
		{"Both Empty", []int{}, []int{}, true},
		{"Nil and Empty", nil, []int{}, true},
		{"Nil and NonEmpty", nil, []int{2, 3}, true},
		{"Both Nil", nil, nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := canBroadcast(tt.a, tt.b); got != tt.want {
				t.Errorf("canBroadcast(%v, %v) = %v, want %v", tt.a, tt.b, got, tt.want)
			}
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
		wantErr   bool
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
		{"Both Nil", nil, nil, []int{}, false},

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
				gotShapeSym := getBroadcastedShape(tt.b, tt.a)
				if !reflect.DeepEqual(gotShapeSym, tt.wantShape) {
					t.Errorf("getBroadcastedShape(%v, %v) [Symmetry check] = %v, want %v", tt.b, tt.a, gotShapeSym, tt.wantShape)
				}
			}
		})
	}
}

func TestTensor_broadcastedIndex(t *testing.T) {

	tests := []struct {
		name          string
		sourceShape   []int
		targetIndices []int
		wantIndex     int
	}{
		{"Source Dim 1 is 1", []int{1, 3}, []int{0, 0}, 0},
		{"Source Dim 1 is 1, Idx 1", []int{1, 3}, []int{0, 1}, 1},
		{"Source Dim 1 is 1, Idx 2", []int{1, 3}, []int{0, 2}, 2},
		{"Source Dim 1 is 1, Target Idx 0 changed", []int{1, 3}, []int{5, 1}, 1},

		{"Source Dim 2 is 1", []int{3, 1}, []int{0, 0}, 0},
		{"Source Dim 2 is 1, Idx 0", []int{3, 1}, []int{1, 0}, 1},
		{"Source Dim 2 is 1, Idx 1", []int{3, 1}, []int{2, 0}, 2},
		{"Source Dim 2 is 1, Target Idx 1 changed", []int{3, 1}, []int{1, 5}, 1},

		{"Source All Dims 1", []int{1, 1}, []int{0, 0}, 0},
		{"Source All Dims 1, Indices Vary", []int{1, 1}, []int{5, 8}, 0},

		{"No Broadcasting Needed", []int{2, 3}, []int{0, 0}, 0},
		{"No Broadcasting Needed 1", []int{2, 3}, []int{0, 2}, 2},
		{"No Broadcasting Needed 2", []int{2, 3}, []int{1, 0}, 3},
		{"No Broadcasting Needed 3", []int{2, 3}, []int{1, 2}, 5},

		{"Mixed 1s and >1s", []int{1, 2, 1, 3}, []int{0, 0, 0, 0}, 0},
		{"Mixed 1s and >1s Idx 1", []int{1, 2, 1, 3}, []int{5, 1, 0, 0}, 3},
		{"Mixed 1s and >1s Idx 2", []int{1, 2, 1, 3}, []int{5, 0, 9, 1}, 1},
		{"Mixed 1s and >1s Idx 3", []int{1, 2, 1, 3}, []int{5, 1, 9, 2}, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := &Tensor{Shape: tt.sourceShape}
			strides := computeStrides(tt.sourceShape)

			gotIndex := tensor.broadcastedIndex(tt.targetIndices, strides)

			if gotIndex != tt.wantIndex {
				t.Errorf("broadcastedIndex(%v, %v) with source shape %v = %d, want %d", tt.targetIndices, strides, tt.sourceShape, gotIndex, tt.wantIndex)
			}
		})
	}
	t.Run("PanicMismatchedLengths", func(t *testing.T) {
		tensor := &Tensor{Shape: []int{2, 3}}
		strides := []int{3, 1}
		badIndices := []int{1, 1, 1}
		checkPanic(t, func() { tensor.broadcastedIndex(badIndices, strides) }, "")

		badStrides := []int{1}
		goodIndices := []int{1, 1}
		checkPanic(t, func() { tensor.broadcastedIndex(goodIndices, badStrides) }, "")
	})

}
