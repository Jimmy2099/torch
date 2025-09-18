package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type Cos struct {
	*OPSNode
	OPSTensor
}

func (m *Cos) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	resultData := make([]float32, len(a.Data))
	for i, val := range a.Data {
		resultData[i] = float32(math.Cos(float64(val)))
	}
	result := tensor.NewTensor(resultData, a.GetShape())
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Cos) Backward(grad *tensor.Tensor) {
	x := m.Children[0].Node.Forward()
	gradData := make([]float32, len(x.Data))
	for i, val := range x.Data {
		gradData[i] = grad.Data[i] * float32(-math.Sin(float64(val)))
	}
	gradInput := tensor.NewTensor(gradData, x.GetShape())
	m.Children[0].Node.Backward(gradInput)
}

func (t *GraphTensor) Cos(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("cos_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewCos(name, t)

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

func NewCos(name string, a *GraphTensor) *Cos {
	return &Cos{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Cos",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Cos) GetOutput() *GraphTensor {
	return m.output
}
