package torch

import "github.com/Jimmy2099/torch/data_store/tensor"

type FlattenLayer struct {
	inputShape []int
}

func (r *FlattenLayer) GetWeights() *tensor.Tensor {
	return tensor.NewEmptyTensor()

}

func (r *FlattenLayer) GetBias() *tensor.Tensor {
	return tensor.NewEmptyTensor()
}

func NewFlattenLayer() *FlattenLayer {
	return &FlattenLayer{}
}

func (f *FlattenLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	f.inputShape = make([]int, len(x.GetShape()))
	copy(f.inputShape, x.GetShape())

	totalElements := 1
	for _, dim := range x.GetShape() {
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
