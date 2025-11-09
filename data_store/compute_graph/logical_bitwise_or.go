package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Or struct {
	*OPSNode
	OPSTensor
}

func (m *Or) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	b := m.Children[1].Node.Forward()

	if len(a.Data) != len(b.Data) {
		panic("tensor sizes must match for OR operation")
	}

	result := a.Or(b)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Or) Backward(grad *tensor.Tensor) {
	return
}

func (t *GraphTensor) Or(other *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("or_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != other.Graph {
		panic("tensors belong to different graphs")
	}
	g := t.Graph

	node := NewOr(name, t, other)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape(t.GetShape())

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewOr(name string, a, b *GraphTensor) *Or {
	return &Or{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Or",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a, b},
		},
	}
}

func (m *Or) GetOutput() *tensor.Tensor {
	return m.output.value
}
