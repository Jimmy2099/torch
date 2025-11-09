package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/node"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type InputNode struct {
	Name string
	//Output *tensor.Tensor
	//Grad   *tensor.Tensor
	output *GraphTensor
}

func (m *InputNode) GetONNXNodeInfo() *node.ONNXNodeInfo {
	return &node.ONNXNodeInfo{
		Name:           "Input",
		ProducedTensor: false,
	}
}

func (n *InputNode) Forward() *tensor.Tensor {
	if n.output == nil {
		return nil
	}

	if n.output.computed {
		return n.output.value
	}

	n.output.computed = true
	return n.output.value
}

func (n *InputNode) Backward(grad *tensor.Tensor) {
	if n.output.grad == nil {
		n.output.grad = tensor.NewTensor(
			make([]float32, len(n.output.value.Data)),
			n.output.value.GetShape(),
		)
	}

	for i := range grad.Data {
		n.output.grad.Data[i] += grad.Data[i]
	}
}

func (n *InputNode) resetGrad() {
	n.output.grad = tensor.NewTensor(make([]float32, len(n.output.value.Data)), n.output.value.GetShape())
}

func (n *InputNode) ResetComputed() {
	n.output.computed = false
}

func (n *InputNode) GetGrad() *tensor.Tensor  { return n.output.grad }
func (n *InputNode) GetName() string          { return n.Name }
func (n *InputNode) GetChildren() []node.Node { return nil }
func (n *InputNode) GetOutput() *tensor.Tensor {
	if n.output == nil {
		return nil
	}
	return n.output.value
}
