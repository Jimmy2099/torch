package raw3d

import (
	"reflect"
	"testing"

	"github.com/Jimmy2099/torch/data_store/tensor"
)

func TestRaw3DToTensor(t *testing.T) {
	makeSeqData := func(n int) []float32 {
		d := make([]float32, n)
		for i := 0; i < n; i++ {
			d[i] = float32(i)
		}
		return d
	}

	tests := []struct {
		name        string
		inputTensor *tensor.Tensor
		r           int
		c           int
		wantErr     bool
		wantShape   []int
		wantData    []float32
	}{
		{
			name:        "Normal Case: 4 Slices (2x2), Single Channel",
			inputTensor: tensor.NewTensor(makeSeqData(16), []int{4, 2, 2, 1}),
			r:           2,
			c:           2,
			wantErr:     false,
			wantShape:   []int{4, 4, 1},
			wantData: []float32{
				0, 1, 4, 5,
				2, 3, 6, 7,
				8, 9, 12, 13,
				10, 11, 14, 15,
			},
		},
		{
			name:        "Normal Case: Multi-Channel (RGB)",
			inputTensor: tensor.NewTensor(makeSeqData(12), []int{2, 1, 2, 3}),
			r:           1,
			c:           2,
			wantErr:     false,
			wantShape:   []int{1, 4, 3},
			wantData: []float32{
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
			},
		},
		{
			name:        "Error: Nil Input",
			inputTensor: nil,
			r:           1,
			c:           1,
			wantErr:     true,
		},
		{
			name:        "Error: Wrong Dimensions (Not 4D)",
			inputTensor: tensor.NewTensor(makeSeqData(8), []int{2, 2, 2}),
			r:           1,
			c:           1,
			wantErr:     true,
		},
		{
			name:        "Error: Grid Size Mismatch",
			inputTensor: tensor.NewTensor(makeSeqData(16), []int{4, 2, 2, 1}),
			r:           2,
			c:           3,
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if (r != nil) != tt.wantErr {
					t.Errorf("Raw3DToTensor() panic = %v, wantPanic %v", r, tt.wantErr)
				}
			}()

			got := Raw3DToTensor(tt.inputTensor, tt.c, tt.r)

			if !tt.wantErr {
				if !reflect.DeepEqual(got.Shape(), tt.wantShape) {
					t.Errorf("Raw3DToTensor() shape = %v, want %v", got.Shape(), tt.wantShape)
				}

				if !reflect.DeepEqual(got.Data, tt.wantData) {
					t.Errorf("Raw3DToTensor() data mismatch.\nGot:  %v\nWant: %v", got.Data, tt.wantData)
				}
			}
		})
	}
}
