package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type Sin struct {
	*OPSNode
	OPSTensor
}

func (m *Sin) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	resultData := make([]float32, len(a.Data))
	for i, val := range a.Data {
		resultData[i] = float32(math.Sin(float64(val)))
	}
	result := tensor.NewTensor(resultData, a.GetShape())
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Sin) Backward(grad *tensor.Tensor) {
	x := m.Children[0].Node.Forward()
	gradData := make([]float32, len(x.Data))
	for i, val := range x.Data {
		gradData[i] = grad.Data[i] * float32(math.Cos(float64(val)))
	}
	gradInput := tensor.NewTensor(gradData, x.GetShape())
	m.Children[0].Node.Backward(gradInput)
}

func (t *GraphTensor) Sin(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("sin_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewSin(name, t)

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

func NewSin(name string, a *GraphTensor) *Sin {
	return &Sin{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Sin",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Sin) GetOutput() *GraphTensor {
	return m.output
}
