package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/node"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type TensorNode struct {
	Name string
	//Output *tensor.Tensor
	//Grad   *tensor.Tensor
	output *GraphTensor
}

func (m *TensorNode) GetONNXNodeInfo() *node.ONNXNodeInfo {
	return &node.ONNXNodeInfo{
		Name:           "Input",
		ProducedTensor: false,
	}
}

func (n *TensorNode) Forward() *tensor.Tensor {
	if n.output == nil {
		return nil
	}

	if n.output.computed {
		return n.output.value
	}

	n.output.computed = true
	return n.output.value
}

func (n *TensorNode) resetGrad() {
	n.output.grad = tensor.NewTensor(make([]float32, len(n.output.value.Data)), n.output.value.GetShape())
}

func (n *TensorNode) ResetComputed() {
	n.output.computed = false
}

func (n *TensorNode) GetGrad() *tensor.Tensor  { return n.output.grad }
func (n *TensorNode) GetName() string          { return n.Name }
func (n *TensorNode) GetChildren() []node.Node { return nil }
func (n *TensorNode) GetOutput() *tensor.Tensor {
	if n.output == nil {
		return nil
	}
	return n.output.value
}
