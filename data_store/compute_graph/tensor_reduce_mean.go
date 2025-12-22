package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ReduceMean struct {
	*OPSNode
	OPSTensor
}

func (m *ReduceMean) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()

	sum := float32(0)
	for _, val := range input.Data {
		sum += val
	}
	mean := sum / float32(len(input.Data))

	result := tensor.NewTensor([]float32{mean}, []int{1})
	m.output.value = result
	m.output.computed = true
	return result
}

func (t *GraphTensor) ReduceMean(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("reduce_mean_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewReduceMean(name, t)

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

func NewReduceMean(name string, a *GraphTensor) *ReduceMean {
	return &ReduceMean{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "ReduceMean",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *ReduceMean) GetOutput() *tensor.Tensor {
	return m.output.value
}
