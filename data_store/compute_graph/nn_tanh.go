package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Tanh struct {
	*OPSNode
	OPSTensor
}

func (m *Tanh) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	result := a.Tanh()
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Tanh) Backward(grad *tensor.Tensor) {
	y := m.output.value
	gradData := make([]float32, len(y.Data))
	for i, val := range y.Data {
		gradData[i] = grad.Data[i] * (1 - val*val)
	}
	gradInput := tensor.NewTensor(gradData, y.GetShape())
	m.Children[0].Node.Backward(gradInput)
}

func (t *GraphTensor) Tanh(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("tanh_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewTanh(name, t)

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

func NewTanh(name string, a *GraphTensor) *Tanh {
	return &Tanh{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Tanh",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Tanh) GetOutput() *tensor.Tensor {
	return m.output.value
}
