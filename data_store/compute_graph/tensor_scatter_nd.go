package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ScatterND struct {
	*OPSNode
	OPSTensor
}

func (m *ScatterND) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	data := m.Children[0].Node.Forward()
	indices := m.Children[1].Node.Forward()
	updates := m.Children[2].Node.Forward()

	result := data.Copy()

	dataShape := data.GetShape()
	indicesShape := indices.GetShape()

	if len(indicesShape) == 2 && indicesShape[1] == 1 && len(dataShape) == 1 {
		for i := 0; i < indicesShape[0]; i++ {
			idx := int(indices.Data[i])
			if idx < dataShape[0] {
				result.Data[idx] = updates.Data[i]
			}
		}
	}

	m.output.value = result
	m.output.computed = true
	return result
}

func (m *ScatterND) Backward(grad *tensor.Tensor) {
	indicesTensor := m.Children[1].Node.GetOutput()
	indicesShape := indicesTensor.Shape()

	numElements := 1
	for _, dim := range indicesShape {
		numElements *= dim
	}

	zeroIndicesGrad := tensor.NewTensor(make([]float32, numElements), indicesShape)
	m.Children[1].Node.Backward(zeroIndicesGrad)

	data := m.Children[0].Node.GetOutput()
	indices := m.Children[1].Node.GetOutput()

	dataGrad := grad.Copy()

	updatesGrad := tensor.NewTensor(make([]float32, indicesShape[0]), []int{indicesShape[0]})

	if len(indicesShape) == 2 && indicesShape[1] == 1 && len(data.Shape()) == 1 {
		for i := 0; i < indicesShape[0]; i++ {
			idx := int(indices.Value().Data[i])
			if idx < len(data.Shape()) {
				updatesGrad.Data[i] = grad.Data[idx]
				dataGrad.Data[idx] = 0
			}
		}
	}

	m.Children[0].Node.Backward(dataGrad)
	m.Children[2].Node.Backward(updatesGrad)
}

func (t *GraphTensor) ScatterND(indices, updates *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("scatternd_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewScatterND(name, t, indices, updates)

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

func NewScatterND(name string, data, indices, updates *GraphTensor) *ScatterND {
	return &ScatterND{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "ScatterND",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{data, indices, updates},
		},
	}
}

func (m *ScatterND) GetOutput() *GraphTensor {
	return m.output
}
