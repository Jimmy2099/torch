package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Sub struct {
	*OPSNode
	OPSTensor
}

func NewSub(name string, a, b *GraphTensor) *Sub {
	return &Sub{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Sub",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a, b},
		},
	}
}

func (m *Sub) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	b := m.Children[1].Node.Forward()

	if len(a.Data) != len(b.Data) {
		panic("tensor sizes must match")
	}

	result := a.Sub(b)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Sub) Backward(grad *tensor.Tensor) {
	aVal := m.Children[0].value
	bVal := m.Children[1].value

	if aVal == nil || bVal == nil || grad == nil {
		panic("nil tensor backward pass")
	}

	gradA := grad.Copy()   // da = grad
	gradB := grad.Negate() // db = -grad

	m.Children[0].Node.Backward(gradA)
	m.Children[1].Node.Backward(gradB)
}

func (t *GraphTensor) Sub(other *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("sub_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != other.Graph {
		panic("tensors belong to different graphs")
	}
	g := t.Graph

	node := NewSub(name, t, other)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape(t.Shape())

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func (m *Sub) GetOutput() *GraphTensor {
	return m.output
}
