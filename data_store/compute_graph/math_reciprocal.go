package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Reciprocal struct {
	*OPSNode
	OPSTensor
}

func (m *Reciprocal) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()

	ones := tensor.NewTensor(make([]float32, len(a.Data)), a.GetShape())
	for i := range ones.Data {
		ones.Data[i] = 1.0
	}

	result := ones.Div(a)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Reciprocal) Backward(grad *tensor.Tensor) {
	aVal := m.Children[0].value

	if aVal == nil || grad == nil {
		panic("nil tensor in reciprocal backward pass")
	}

	ones := tensor.NewTensor(make([]float32, len(aVal.Data)), aVal.GetShape())
	for i := range ones.Data {
		ones.Data[i] = 1.0
	}

	aSquared := aVal.Copy().Mul(aVal)
	negOneOverASquared := ones.Div(aSquared).Negate()
	gradA := negOneOverASquared.Mul(grad)

	m.Children[0].Node.Backward(gradA)
}

func (t *GraphTensor) Reciprocal(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("reciprocal_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph

	node := NewReciprocal(name, t)

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

func NewReciprocal(name string, a *GraphTensor) *Reciprocal {
	return &Reciprocal{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Reciprocal",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Reciprocal) GetOutput() *GraphTensor {
	return m.output
}
