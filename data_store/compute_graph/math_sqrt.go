package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Sqrt struct {
	*OPSNode
	OPSTensor
}

func (m *Sqrt) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}
	input := m.Children[0].Node.Forward()
	out := input.Clone().Sqrt()
	m.output.value = tensor.NewTensor(out.Data, input.GetShape())
	m.output.computed = true
	return m.output.value
}

func (m *Sqrt) Backward(grad *tensor.Tensor) {
	outputVal := m.output.value
	if outputVal == nil || grad == nil {
		panic("nil tensor in sqrt backward pass")
	}

	gradData := make([]float32, len(outputVal.Data))
	for i, out := range outputVal.Data {
		gradData[i] = grad.Data[i] / (2 * out)
	}
	gradInput := tensor.NewTensor(gradData, outputVal.GetShape())
	m.Children[0].Node.Backward(gradInput)
}

func (t *GraphTensor) Sqrt(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("sqrt_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph

	node := NewSqrt(name, t)

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

func NewSqrt(name string, a *GraphTensor) *Sqrt {
	return &Sqrt{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Sqrt",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Sqrt) GetOutput() *GraphTensor {
	return m.output
}
