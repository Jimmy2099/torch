package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Not struct {
	*OPSNode
	OPSTensor
}

func (m *Not) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()

	result := a.Not()
	m.output.value = result
	m.output.computed = true
	return result
}

func (t *GraphTensor) Not(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("not_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph

	node := NewNot(name, t)

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

func NewNot(name string, a *GraphTensor) *Not {
	return &Not{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "And",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Not) GetOutput() *tensor.Tensor {
	return m.output.value
}
