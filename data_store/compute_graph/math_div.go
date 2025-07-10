package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Div struct {
	OPS
}

func (m *Div) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	b := m.Children[1].Node.Forward()

	if len(a.Data) != len(b.Data) {
		panic("tensor sizes must match for division")
	}

	result := a.Div(b)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Div) Backward(grad *tensor.Tensor) {
	aVal := m.Children[0].value
	bVal := m.Children[1].value

	if aVal == nil || bVal == nil || grad == nil {
		panic("nil tensor in division backward pass")
	}

	// Create a tensor of ones with the same shape as bVal
	ones := tensor.NewTensor(make([]float32, len(bVal.Data)), bVal.GetShape())
	for i := range ones.Data {
		ones.Data[i] = 1.0
	}

	// Gradient for numerator: da = grad / b = grad * (1/b)
	gradA := ones.Copy().Div(bVal).Mul(grad)

	// Gradient for denominator: db = -grad * a / (b^2) = -grad * a * (1/b^2)
	bSquared := bVal.Copy().Mul(bVal)
	oneOverBSquared := ones.Copy().Div(bSquared)
	gradB := oneOverBSquared.Mul(aVal).Mul(grad).Negate()

	m.Children[0].Node.Backward(gradA)
	m.Children[1].Node.Backward(gradB)
}

func (t *GraphTensor) Div(other *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("div_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != other.Graph {
		panic("tensors belong to different graphs")
	}
	g := t.Graph

	node := NewDiv(name, t, other)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Shape: t.Shape,
		Graph: g,
		Node:  node,
	}

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewDiv(name string, a, b *GraphTensor) *Div {
	return &Div{
		OPS{
			Name:     name,
			Children: []*GraphTensor{a, b},
		},
	}
}
