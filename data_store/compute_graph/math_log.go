package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math"
)

type Log struct {
	*OPSNode
	OPSTensor
}

func (m *Log) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}
	input := m.Children[0].Node.Forward()
	data := make([]float32, len(input.Data))
	for i, v := range input.Data {
		if v <= 0 {
			panic("log input must be positive")
		}
		data[i] = float32(math.Log(float64(v)))
	}
	m.output.value = tensor.NewTensor(data, input.GetShape())
	m.output.computed = true
	return m.output.value
}

func (t *GraphTensor) Log(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("log_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph

	node := NewLog(name, t)

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

func NewLog(name string, a *GraphTensor) *Log {
	return &Log{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Log",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{Name: name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *Log) GetOutput() *tensor.Tensor {
	return m.output.value
}
