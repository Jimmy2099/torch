package node

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Node interface {
	Forward() *tensor.Tensor
	Backward(grad *tensor.Tensor)
	GetName() string
	ResetComputed()

	GetONNXNodeInfo() *ONNXNodeInfo
	GetChildren() []Node
	GetOutput() *tensor.Tensor
}

type ONNXNodeInfo struct {
	Name           string
	ProducedTensor bool
}
