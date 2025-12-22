package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ReduceMax struct {
	*OPSNode
	OPSTensor
}

func (m *ReduceMax) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()

	maxVal := input.Data[0]
	for _, val := range input.Data {
		if val > maxVal {
			maxVal = val
		}
	}

	result := tensor.NewTensor([]float32{maxVal}, []int{1})
	m.output.value = result
	m.output.computed = true
	return result
}

func (t *GraphTensor) ReduceMax(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("reduce_max_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewReduceMax(name, t)

	outputShape := []int{1}
	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape(outputShape)

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewReduceMax(name string, a *GraphTensor) *ReduceMax {
	return &ReduceMax{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "ReduceMax",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *ReduceMax) GetOutput() *tensor.Tensor {
	return m.output.value
}
