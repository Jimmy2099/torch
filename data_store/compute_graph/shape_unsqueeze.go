package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Unsqueeze struct {
	*OPSNode
	OPSTensor
	originalShape []int
	axis          int
}

func (u *Unsqueeze) Forward() *tensor.Tensor {
	if u.output.computed {
		return u.output.value
	}

	input := u.Children[0].Node.Forward()
	u.originalShape = input.GetShape()

	newShape := make([]int, len(u.originalShape)+1)
	copy(newShape[:u.axis], u.originalShape[:u.axis])
	newShape[u.axis] = 1
	copy(newShape[u.axis+1:], u.originalShape[u.axis:])

	result := input.Reshape(newShape)
	u.output.value = result
	u.output.computed = true
	return result
}

func (u *Unsqueeze) Backward(grad *tensor.Tensor) {
	if grad == nil {
		panic("nil gradient in unsqueeze backward pass")
	}
	gradInput := grad.Reshape(u.originalShape)
	u.Children[0].Node.Backward(gradInput)
}

func (t *GraphTensor) Unsqueeze(axis int, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("unsqueeze_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewUnsqueeze(name, t, &GraphTensor{
		value: tensor.NewTensor([]float32{float32(axis)}, []int{1}),
	})

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape(node.Children[0].GetShape())

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewUnsqueeze(name string, a *GraphTensor, axis *GraphTensor) *Unsqueeze {
	v := 0
	if axis != nil && axis.value != nil && axis.value.Data != nil {
		v = int(axis.value.Data[0])
	}

	return &Unsqueeze{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Unsqueeze",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
		axis: v, //TODO
	}
}

func (m *Unsqueeze) GetOutput() *tensor.Tensor {
	return m.output.value
}
