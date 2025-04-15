package torch

import "github.com/Jimmy2099/torch/data_store/tensor"

type FlattenLayer struct {
	inputShape []int
}

func (r *FlattenLayer) GetWeights() *tensor.Tensor {
	return &tensor.Tensor{
		Data:  make([]float32, 0),
		Shape: make([]int, 0),
	}
}

func (r *FlattenLayer) GetBias() *tensor.Tensor {
	return &tensor.Tensor{
		Data:  make([]float32, 0),
		Shape: make([]int, 0),
	}
}

func NewFlattenLayer() *FlattenLayer {
	return &FlattenLayer{}
}

func (f *FlattenLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	f.inputShape = make([]int, len(x.Shape))
	copy(f.inputShape, x.Shape)

	totalElements := 1
	for _, dim := range x.Shape {
		totalElements *= dim
	}

	return x.Reshape([]int{totalElements})
}

func (f *FlattenLayer) Backward(dout *tensor.Tensor) *tensor.Tensor {
	if len(f.inputShape) == 0 {
		panic("FlattenLayer: input shape not recorded during forward pass")
	}
	return dout.Reshape(f.inputShape)
}
