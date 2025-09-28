package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type Floor struct {
	*OPSNode
	OPSTensor
}

func (m *Floor) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	data := make([]float32, len(a.Data))
	for i, v := range a.Data {
		data[i] = float32(math.Floor(float64(v)))
	}

	result := tensor.NewTensor(data, a.GetShape())
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Floor) Backward(grad *tensor.Tensor) {
	gradInput := tensor.NewTensor(make([]float32, len(grad.Data)), grad.GetShape())
	m.Children[0].Node.Backward(gradInput)
}

func (t *GraphTensor) Floor(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("floor_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph

	node := NewFloor(name, t)

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

func NewFloor(name string, a *GraphTensor) *Floor {
	return &Floor{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Floor",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{Name: name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Floor) GetOutput() *GraphTensor {
	return m.output
}
