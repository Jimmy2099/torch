package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ReduceMin struct {
	*OPSNode
	OPSTensor
}

func (m *ReduceMin) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()

	minVal := input.Data[0]
	for _, val := range input.Data {
		if val < minVal {
			minVal = val
		}
	}

	result := tensor.NewTensor([]float32{minVal}, []int{1})
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *ReduceMin) Backward(grad *tensor.Tensor) {
	input := m.Children[0].Node.Forward()

	minVal := input.Data[0]
	minIndices := []int{0}
	for i, val := range input.Data {
		if val < minVal {
			minVal = val
			minIndices = []int{i}
		} else if val == minVal {
			minIndices = append(minIndices, i)
		}
	}

	gradData := make([]float32, len(input.Data))
	gradPerMin := grad.Data[0] / float32(len(minIndices))
	for _, idx := range minIndices {
		gradData[idx] = gradPerMin
	}

	gradInput := tensor.NewTensor(gradData, input.GetShape())
	m.Children[0].Node.Backward(gradInput)
}

func (t *GraphTensor) ReduceMin(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("reduce_min_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewReduceMin(name, t)

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

func NewReduceMin(name string, a *GraphTensor) *ReduceMin {
	return &ReduceMin{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "ReduceMin",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *ReduceMin) GetOutput() *GraphTensor {
	return m.output
}
