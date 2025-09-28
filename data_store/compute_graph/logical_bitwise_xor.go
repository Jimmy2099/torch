package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Xor struct {
	*OPSNode
	OPSTensor
}

func (m *Xor) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	b := m.Children[1].Node.Forward()

	if len(a.Data) != len(b.Data) {
		panic("tensor sizes must match for XOR operation")
	}

	result := a.Xor(b)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Xor) Backward(grad *tensor.Tensor) {
	aVal := m.Children[0].value
	bVal := m.Children[1].value

	if aVal == nil || bVal == nil || grad == nil {
		panic("nil tensor in XOR backward pass")
	}

	gradA := tensor.NewTensor(make([]float32, len(aVal.Data)), aVal.GetShape())
	for i := range gradA.Data {
		gradA.Data[i] = 1.0 - 2*bVal.Data[i]
	}
	gradA = gradA.Mul(grad)

	gradB := tensor.NewTensor(make([]float32, len(bVal.Data)), bVal.GetShape())
	for i := range gradB.Data {
		gradB.Data[i] = 1.0 - 2*aVal.Data[i]
	}
	gradB = gradB.Mul(grad)

	m.Children[0].Node.Backward(gradA)
	m.Children[1].Node.Backward(gradB)
}

func (t *GraphTensor) Xor(other *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("xor_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != other.Graph {
		panic("tensors belong to different graphs")
	}
	g := t.Graph

	node := NewXor(name, t, other)

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

func NewXor(name string, a, b *GraphTensor) *Xor {
	return &Xor{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Xor",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a, b},
		},
	}
}

func (m *Xor) GetOutput() *GraphTensor {
	return m.output
}
