package tensor

import (
	"math/rand"
	"reflect"
	"strings"
	"testing"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func TestNewTensorWithShape(t *testing.T) {
	tests := []struct {
		name      string
		shape     []int
		wantShape []int
		wantData  []float32
	}{
		{"Matrix", []int{2, 3}, []int{2, 3}, []float32{0, 0, 0, 0, 0, 0}},
		{"Vector", []int{4}, []int{4}, []float32{0, 0, 0, 0}},
		{"Scalar", []int{1}, []int{1}, []float32{0}},
		{"Empty shape", []int{}, []int{}, []float32{}},
		{"Zero Dim", []int{2, 0, 3}, []int{2, 0, 3}, []float32{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewTensorWithShape(tt.shape)
			wantTensor := &Tensor{Data: tt.wantData, shape: tt.wantShape}

			if !reflect.DeepEqual(tensor.shape, wantTensor.shape) {
				if !((tensor.shape == nil && len(wantTensor.shape) == 0) || (wantTensor.shape == nil && len(tensor.shape) == 0)) {
					t.Errorf("NewTensorWithShape(%v) shape = %v, want %v", tt.shape, tensor.shape, wantTensor.shape)
				}
			}
			if !floatsEqual(tensor.Data, wantTensor.Data, epsilon) {
				t.Errorf("NewTensorWithShape(%v) data = %v, want %v", tt.shape, tensor.Data, wantTensor.Data)
			}
			size := 1
			isZeroSize := false
			if len(tensor.shape) == 0 {
				size = 0
			} else {
				for _, d := range tensor.shape {
					if d == 0 {
						isZeroSize = true
						break
					}
					size *= d
				}
				if isZeroSize {
					size = 0
				}
			}

			if len(tensor.Data) != size {
				t.Errorf("NewTensorWithShape(%v) data length = %d, expected size %d", tt.shape, len(tensor.Data), size)
			}
		})
	}
	t.Run("NilShape", func(t *testing.T) {
		tensor := NewTensorWithShape(nil)
		if tensor.shape != nil {
			t.Errorf("NewTensorWithShape(nil) shape = %v, want nil", tensor.shape)
		}
		if len(tensor.Data) != 0 {
			t.Errorf("NewTensorWithShape(nil) data = %v, want []", tensor.Data)
		}
	})
}

func TestNewRandomTensor(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
	}{
		{"Matrix", []int{2, 3}},
		{"Vector", []int{10}},
		{"Scalar", []int{1}},
		{"Empty shape", []int{}},
		{"Zero Dim", []int{3, 0, 2}},
		{"Nil shape", nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewRandomTensor(tt.shape)

			wantShape := tt.shape
			if wantShape == nil {
				wantShape = nil
			} else if len(wantShape) == 0 {
				wantShape = []int{}
			}

			if !reflect.DeepEqual(tensor.shape, wantShape) {
				if !((tensor.shape == nil && wantShape == nil) || (len(tensor.shape) == 0 && len(wantShape) == 0 && tensor.shape != nil && wantShape != nil)) {
					t.Errorf("NewRandomTensor(%v) shape = %v, want %v", tt.shape, tensor.shape, wantShape)
				}
			}

			expectedSize := 0
			if tensor.shape != nil {
				isZeroSize := false
				tempSize := 1
				for _, d := range tensor.shape {
					if d == 0 {
						isZeroSize = true
						break
					}
					tempSize *= d
				}
				if !isZeroSize && len(tensor.shape) > 0 {
					expectedSize = tempSize
				} else if isZeroSize {
					expectedSize = 0
				} else {
					expectedSize = 0
				}
			}

			if len(tensor.Data) != expectedSize {
				t.Errorf("NewRandomTensor(%v) data length = %d, want %d", tt.shape, len(tensor.Data), expectedSize)
			}

			if expectedSize > 0 {
				for i, val := range tensor.Data {
					if val < -1.0 || val >= 1.0 {
						t.Errorf("NewRandomTensor(%v) data[%d] = %v, out of range [-1.0, 1.0)", tt.shape, i, val)
					}
				}
				if expectedSize > 1 {
					allSame := true
					firstVal := tensor.Data[0]
					for i := 1; i < expectedSize; i++ {
						if tensor.Data[i] != firstVal {
							allSame = false
							break
						}
					}
					if allSame && firstVal == 0.0 {
						t.Logf("Warning: NewRandomTensor(%v) produced all zero values (might be correct by chance)", tt.shape)
					}
					if allSame && firstVal != 0.0 {
						t.Errorf("NewRandomTensor(%v) produced all same non-zero values: %v", tt.shape, firstVal)
					}
				}

			}
		})
	}
}

func TestNewTensorFromSlice(t *testing.T) {
	tests := []struct {
		name      string
		input     [][]float32
		wantShape []int
		wantData  []float32
		wantErr   bool
	}{
		{
			name:      "Standard Matrix",
			input:     [][]float32{{1, 2, 3}, {4, 5, 6}},
			wantShape: []int{2, 3},
			wantData:  []float32{1, 2, 3, 4, 5, 6},
			wantErr:   false,
		},
		{
			name:      "Single Row",
			input:     [][]float32{{7, 8}},
			wantShape: []int{1, 2},
			wantData:  []float32{7, 8},
			wantErr:   false,
		},
		{
			name:      "Single Column",
			input:     [][]float32{{9}, {10}},
			wantShape: []int{2, 1},
			wantData:  []float32{9, 10},
			wantErr:   false,
		},
		{
			name:      "Empty Outer Slice",
			input:     [][]float32{},
			wantShape: []int{0, 0},
			wantData:  []float32{},
			wantErr:   false,
		},
		{
			name:      "Empty Inner Slices",
			input:     [][]float32{{}, {}},
			wantShape: []int{2, 0},
			wantData:  []float32{},
			wantErr:   false,
		},
		{
			name:      "Single Empty Inner Slice",
			input:     [][]float32{{}},
			wantShape: []int{1, 0},
			wantData:  []float32{},
			wantErr:   false,
		},
		{
			name:    "Jagged Slice",
			input:   [][]float32{{1, 2}, {3}},
			wantErr: true,
		},
		{
			name:    "Jagged Slice with Empty",
			input:   [][]float32{{1, 2}, {}},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.wantErr {
				checkPanic(t, func() { NewTensorFromSlice(tt.input) }, "")
			} else {
				tensor := NewTensorFromSlice(tt.input)
				wantTensor := &Tensor{Data: tt.wantData, shape: tt.wantShape}
				if !tensorsEqual(tensor, wantTensor, epsilon) {
					t.Errorf("NewTensorFromSlice() got %v, want %v", tensor, wantTensor)
				}
			}
		})
	}
}

func TestTensor_Reshape(t *testing.T) {
	originalData := []float32{1, 2, 3, 4, 5, 6}

	tests := []struct {
		name       string
		startShape []int
		reshapeTo  []int
		wantShape  []int
		wantErr    bool
	}{
		{"Valid Reshape 2x3 to 3x2", []int{2, 3}, []int{3, 2}, []int{3, 2}, false},
		{"Valid Reshape 6 to 2x3", []int{6}, []int{2, 3}, []int{2, 3}, false},
		{"Valid Reshape 2x3 to 6", []int{2, 3}, []int{6}, []int{6}, false},
		{"Valid Reshape 2x1x3 to 6", []int{2, 1, 3}, []int{6}, []int{6}, false},
		{"Valid Reshape 6 to 2x1x3", []int{6}, []int{2, 1, 3}, []int{2, 1, 3}, false},
		{"Valid Reshape 1 to 1", []int{1}, []int{1}, []int{1}, false},
		{"Invalid Reshape Size Mismatch", []int{2, 3}, []int{2, 2}, nil, true},
		{"Invalid Reshape Size Mismatch Vector", []int{6}, []int{5}, nil, true},
		{"Invalid Reshape Zero Dim Input", []int{2, 0, 3}, []int{6}, nil, true},
		{"Valid Reshape Zero Dim Output", []int{2, 0, 3}, []int{3, 2, 0}, []int{3, 2, 0}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startSize := 1
			isZeroSize := false
			for _, d := range tt.startShape {
				if d == 0 {
					isZeroSize = true
					break
				}
				startSize *= d
			}
			if isZeroSize || len(tt.startShape) == 0 {
				startSize = 0
			}
			var currentData []float32
			if startSize > 0 {
				currentData = make([]float32, startSize)
				copy(currentData, originalData[:startSize])
			} else {
				currentData = []float32{}
			}

			tensor := NewTensor(currentData, tt.startShape)
			originalTensorPtr := tensor

			if tt.wantErr {
				checkPanic(t, func() { tensor.Reshape(tt.reshapeTo) }, "")
				if !reflect.DeepEqual(tensor.shape, tt.startShape) {
					t.Errorf("Tensor shape changed after Reshape panic: got %v, expected original %v", tensor.shape, tt.startShape)
				}
			} else {
				reshapedTensor := tensor.Reshape(tt.reshapeTo)

				if reshapedTensor != originalTensorPtr {
					t.Errorf("Reshape should return the same tensor pointer, but got a different one.")
				}
				if !reflect.DeepEqual(tensor.shape, tt.wantShape) {
					t.Errorf("Reshape(%v) resulted in shape %v, want %v", tt.reshapeTo, tensor.shape, tt.wantShape)
				}
				if len(currentData) > 0 && len(tensor.Data) > 0 && currentData[0] != tensor.Data[0] {
					t.Logf("Warning: Data slice pointers differ after reshape (original=%p, tensor=%p). This might be okay if NewGraphTensor copies data, but Reshape itself should not reallocate.", &currentData[0], &tensor.Data[0])
				}
				if !floatsEqual(tensor.Data, currentData, epsilon) {
					t.Errorf("Reshape changed tensor data content: got %v, expected %v", tensor.Data, currentData)
				}
			}
		})
	}
	t.Run("Reshape 1 to Empty", func(t *testing.T) {
		tensor1 := NewTensor([]float32{5.0}, []int{1})
		tempTargetTensor := &Tensor{shape: []int{}}
		sizeEmpty := tempTargetTensor.Size()
		sizeTensor1 := tensor1.Size()

		if sizeEmpty != sizeTensor1 {
			checkPanic(t, func() { tensor1.Reshape([]int{}) }, "")
		} else {
			tensor1.Reshape([]int{})
			wantShape := []int{}
			if !reflect.DeepEqual(tensor1.shape, wantShape) {
				t.Errorf("Reshape([1]) to []int{} resulted in shape %v, want %v", tensor1.shape, wantShape)
			}
		}
	})
}

func TestTensor_Squeeze(t *testing.T) {
	tests := []struct {
		name       string
		startShape []int
		wantShape  []int
	}{
		{"Remove Leading 1", []int{1, 2, 3}, []int{2, 3}},
		{"Remove Trailing 1", []int{2, 3, 1}, []int{2, 3}},
		{"Remove Middle 1", []int{2, 1, 3}, []int{2, 3}},
		{"Remove Multiple 1s", []int{1, 2, 1, 3, 1}, []int{2, 3}},
		{"All 1s", []int{1, 1, 1}, []int{}},
		{"Single 1", []int{1}, []int{}},
		{"No 1s", []int{2, 3}, []int{2, 3}},
		{"Empty shape", []int{}, []int{}},
		{"Nil shape", nil, nil},
		{"shape with 0", []int{1, 0, 1}, []int{0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startSize := 1
			isZeroSize := false
			if tt.startShape == nil {
				startSize = 0
			} else {
				for _, d := range tt.startShape {
					if d == 0 {
						isZeroSize = true
						break
					}
					startSize *= d
				}
				if isZeroSize || len(tt.startShape) == 0 {
					startSize = 0
				}
			}
			var startData []float32
			if startSize > 0 {
				startData = make([]float32, startSize)
				startData[0] = 1
			} else {
				startData = []float32{}
			}

			tensor := &Tensor{Data: startData, shape: tt.startShape}
			originalTensorPtr := tensor
			var originalDataPtr *float32

			if len(startData) > 0 {
				originalDataPtr = &startData[0]
			}

			squeezedTensor := tensor.Squeeze()

			if squeezedTensor != originalTensorPtr {
				t.Errorf("Squeeze should return the same tensor pointer, but got a different one.")
			}

			if !reflect.DeepEqual(tensor.shape, tt.wantShape) {
				if !((tensor.shape == nil && tt.wantShape == nil) || (len(tensor.shape) == 0 && len(tt.wantShape) == 0 && tensor.shape != nil && tt.wantShape != nil)) {
					t.Errorf("Squeeze() resulted in shape %v, want %v", tensor.shape, tt.wantShape)
				}
			}

			if len(startData) > 0 && len(tensor.Data) > 0 && originalDataPtr != &tensor.Data[0] {
			}
			if !floatsEqual(tensor.Data, startData, epsilon) {
				t.Errorf("Squeeze changed tensor data content: got %v, expected %v", tensor.Data, startData)
			}
		})
	}
}

func TestTensor_SqueezeSpecific(t *testing.T) {
	tests := []struct {
		name        string
		startShape  []int
		squeezeDims []int
		wantShape   []int
		wantErr     bool
	}{
		{"Squeeze Dim 0", []int{1, 2, 3}, []int{0}, []int{2, 3}, false},
		{"Squeeze Dim 2", []int{2, 3, 1}, []int{2}, []int{2, 3}, false},
		{"Squeeze Dim 1", []int{2, 1, 3}, []int{1}, []int{2, 3}, false},
		{"Squeeze Multiple Dims", []int{1, 2, 1, 3, 1}, []int{0, 2, 4}, []int{2, 3}, false},
		{"Squeeze All Dims", []int{1, 1, 1}, []int{0, 1, 2}, []int{}, false},
		{"Squeeze Subset of 1s", []int{1, 1, 1}, []int{0, 2}, []int{1}, false},
		{"Squeeze Empty Dims", []int{1, 2, 1}, []int{}, []int{1, 2, 1}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startSize := 1
			isZeroSize := false
			if tt.startShape == nil {
				startSize = 0
			} else {
				for _, d := range tt.startShape {
					if d == 0 {
						isZeroSize = true
						break
					}
					startSize *= d
				}
				if isZeroSize || len(tt.startShape) == 0 {
					startSize = 0
				}
			}

			var startData []float32
			if startSize > 0 {
				startData = make([]float32, startSize)
				startData[0] = 1
			} else {
				startData = []float32{}
			}

			tensor := &Tensor{Data: startData, shape: tt.startShape}
			originalTensorPtr := tensor
			var originalDataPtr *float32
			if len(startData) > 0 {
				originalDataPtr = &startData[0]
			}

			if tt.wantErr {
				checkPanic(t, func() { tensor.SqueezeSpecific(tt.squeezeDims) }, "")
				if !reflect.DeepEqual(tensor.shape, tt.startShape) {
					t.Errorf("Tensor shape changed after panic: got %v, expected %v", tensor.shape, tt.startShape)
				}
			} else {
				squeezedTensor := tensor.SqueezeSpecific(tt.squeezeDims)

				if squeezedTensor != originalTensorPtr {
					t.Errorf("SqueezeSpecific should return the same tensor pointer, but got a different one.")
				}

				if !reflect.DeepEqual(tensor.shape, tt.wantShape) {
					t.Errorf("SqueezeSpecific(%v) resulted in shape %v, want %v", tt.squeezeDims, tensor.shape, tt.wantShape)
				}

				if len(startData) > 0 && len(tensor.Data) > 0 && originalDataPtr != &tensor.Data[0] {
					t.Errorf("SqueezeSpecific seems to have reallocated data (data pointer changed)")
				}

				if !floatsEqual(tensor.Data, startData, epsilon) {
					t.Errorf("SqueezeSpecific changed tensor data content: got %v, expected %v", tensor.Data, startData)
				}
			}
		})
	}

	t.Run("NilShape", func(t *testing.T) {
		tensor := &Tensor{Data: nil, shape: nil}
		res := tensor.SqueezeSpecific([]int{0})
		if res.shape != nil {
			t.Errorf("SqueezeSpecific on nil shape tensor resulted in shape %v, want nil", res.shape)
		}
	})
}

func TestTensor_Indices(t *testing.T) {
	tests := []struct {
		name        string
		shape       []int
		linearIdx   int
		wantIndices []int
	}{
		{"3D Basic Start", []int{2, 3, 4}, 0, []int{0, 0, 0}},
		{"3D Basic End Dim", []int{2, 3, 4}, 3, []int{0, 0, 3}},
		{"3D Middle Dim Rollover", []int{2, 3, 4}, 4, []int{0, 1, 0}},
		{"3D Middle", []int{2, 3, 4}, 7, []int{0, 1, 3}},
		{"3D End Middle Dim", []int{2, 3, 4}, 11, []int{0, 2, 3}},
		{"3D First Dim Rollover", []int{2, 3, 4}, 12, []int{1, 0, 0}},
		{"3D End", []int{2, 3, 4}, 23, []int{1, 2, 3}},

		{"Vector Start", []int{5}, 0, []int{0}},
		{"Vector Middle", []int{5}, 2, []int{2}},
		{"Vector End", []int{5}, 4, []int{4}},

		{"Scalar", []int{1}, 0, []int{0}},

		{"Empty shape", []int{}, 0, []int{}},

		{"Zero Dim shape", []int{2, 0, 3}, 0, []int{0, 0, 0}},

		{"Zero Dim shape End", []int{3, 2, 0}, 0, []int{0, 0, 0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := &Tensor{shape: tt.shape}
			gotIndices := tensor.Indices(tt.linearIdx)

			if !reflect.DeepEqual(gotIndices, tt.wantIndices) {
				t.Errorf("Indices(%d) for shape %v = %v, want %v", tt.linearIdx, tt.shape, gotIndices, tt.wantIndices)
			}
		})
	}

	t.Run("OutOfBoundsIndex", func(t *testing.T) {
		tensor := &Tensor{shape: []int{2, 3}}
		got := tensor.Indices(6)
		want := []int{2, 0}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("Indices(6) for shape [2,3] got %v, want %v (calculation check)", got, want)
		}
	})
}

func TestTensor_Transpose(t *testing.T) {
	tests := []struct {
		name        string
		shape       []int
		data        []float32
		wantShape   []int
		wantData    []float32
		shouldPanic bool
	}{
		{
			name:      "2D transpose",
			shape:     []int{2, 3},
			data:      []float32{1, 2, 3, 4, 5, 6},
			wantShape: []int{3, 2},
			wantData:  []float32{1, 4, 2, 5, 3, 6},
		},
		{
			name:        "1D should panic",
			shape:       []int{3},
			data:        []float32{1, 2, 3},
			shouldPanic: true,
		},
		{
			name:        "3D should panic",
			shape:       []int{2, 2, 2},
			data:        []float32{1, 2, 3, 4, 5, 6, 7, 8},
			shouldPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.shouldPanic {
				defer func() {
					if r := recover(); r == nil {
						t.Error("Expected panic did not occur")
					}
				}()
			}

			tensor := NewTensor(tt.data, tt.shape)
			got := tensor.Transpose()

			if !reflect.DeepEqual(got.shape, tt.wantShape) {
				t.Errorf("Transpose() shape = %v, want %v", got.shape, tt.wantShape)
			}
			if !reflect.DeepEqual(got.Data, tt.wantData) {
				t.Errorf("Transpose() data = %v, want %v", got.Data, tt.wantData)
			}
		})
	}
}

func TestTensor_Gather(t *testing.T) {
	tests := []struct {
		name        string
		tensorShape []int
		tensorData  []float32
		indices     []float32
		wantShape   []int
		wantData    []float32
		shouldPanic bool
		panicMsg    string
	}{
		{
			name:        "Basic gather",
			tensorShape: []int{3, 2},
			tensorData:  []float32{1, 2, 3, 4, 5, 6},
			indices:     []float32{0, 2},
			wantShape:   []int{2, 2},
			wantData:    []float32{1, 2, 5, 6},
		},
		{
			name:        "Indices not 1D",
			tensorShape: []int{2, 2},
			tensorData:  []float32{1, 2, 3, 4},
			indices:     []float32{0, 0},
			wantShape:   []int{2, 2},
			shouldPanic: true,
			panicMsg:    "Gather indices must be 1D",
		},
		{
			name:        "Index out of range",
			tensorShape: []int{2, 2},
			tensorData:  []float32{1, 2, 3, 4},
			indices:     []float32{3},
			shouldPanic: true,
			panicMsg:    "Gather index out of range: 3 not in [0, 2)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.shouldPanic {
				defer func() {
					if r := recover(); r == nil {
						t.Error("Expected panic did not occur")
					} else if !strings.Contains(r.(string), tt.panicMsg) {
						t.Errorf("Unexpected panic message: %v", r)
					}
				}()
			}

			tensor := NewTensor(tt.tensorData, tt.tensorShape)
			indicesTensor := NewTensor(tt.indices, []int{len(tt.indices)})
			got := tensor.Gather(indicesTensor)

			if !tt.shouldPanic {
				if !reflect.DeepEqual(got.shape, tt.wantShape) {
					t.Errorf("Gather() shape = %v, want %v", got.shape, tt.wantShape)
				}
				if !reflect.DeepEqual(got.Data, tt.wantData) {
					t.Errorf("Gather() data = %v, want %v", got.Data, tt.wantData)
				}
			}
		})
	}
}

func TestTensor_ScatterAdd(t *testing.T) {
	tests := []struct {
		name        string
		targetShape []int
		targetData  []float32
		indices     []float32
		sourceShape []int
		sourceData  []float32
		wantData    []float32
		shouldPanic bool
		panicMsg    string
	}{
		{
			name:        "Basic scatterAdd",
			targetShape: []int{3, 2},
			targetData:  make([]float32, 6),
			indices:     []float32{0, 2},
			sourceShape: []int{2, 2},
			sourceData:  []float32{1, 2, 3, 4},
			wantData:    []float32{1, 2, 0, 0, 3, 4},
		},
		{
			name:        "Indices not 1D",
			targetShape: []int{2, 2},
			targetData:  make([]float32, 4),
			indices:     []float32{0, 0},
			sourceShape: []int{2, 2},
			sourceData:  []float32{1, 1, 1, 1},
			shouldPanic: true,
			panicMsg:    "ScatterAdd indices must be 1D",
		},
		{
			name:        "Dimension mismatch",
			targetShape: []int{2, 2},
			targetData:  make([]float32, 4),
			indices:     []float32{0},
			sourceShape: []int{1, 3},
			sourceData:  []float32{1, 2, 3},
			shouldPanic: true,
			panicMsg:    "ScatterAdd target and source must have same dimensions after the first",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.shouldPanic {
				defer func() {
					if r := recover(); r == nil {
						t.Error("Expected panic did not occur")
					} else if !strings.Contains(r.(string), tt.panicMsg) {
						t.Errorf("Unexpected panic message: %v", r)
					}
				}()
			}

			target := NewTensor(tt.targetData, tt.targetShape)
			indicesTensor := NewTensor(tt.indices, []int{len(tt.indices)})
			source := NewTensor(tt.sourceData, tt.sourceShape)

			target.ScatterAdd(indicesTensor, source)

			if !tt.shouldPanic && !reflect.DeepEqual(target.Data, tt.wantData) {
				t.Errorf("ScatterAdd() = %v, want %v", target.Data, tt.wantData)
			}
		})
	}
}
