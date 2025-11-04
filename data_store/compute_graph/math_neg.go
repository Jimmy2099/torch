package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Neg struct {
	*OPSNode
	OPSTensor
}

func (m *Neg) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}
	input := m.Children[0].Node.Forward()
	m.output.value = input.Negate()
	m.output.computed = true
	return m.output.value
}

func (m *Neg) Backward(grad *tensor.Tensor) {
	m.Children[0].Node.Backward(grad.Negate())
}

func (t *GraphTensor) Neg(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("neg_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph

	node := NewNeg(name, t)

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

func NewNeg(name string, a *GraphTensor) *Neg {
	return &Neg{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Neg",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Neg) GetOutput() *tensor.Tensor {
	return m.output.value
}
