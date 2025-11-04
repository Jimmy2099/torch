package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type ShapeOp struct {
	*OPSNode
	OPSTensor
}

func (m *ShapeOp) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	shape := a.GetShape()
	shapeData := make([]float32, len(shape))
	for i, dim := range shape {
		shapeData[i] = float32(dim)
	}
	result := tensor.NewTensor(shapeData, []int{len(shape)})
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *ShapeOp) Backward(grad *tensor.Tensor) {
	inputShape := m.Children[0].Node.GetOutput().GetShape()

	numElements := 1
	for _, dim := range inputShape {
		numElements *= dim
	}

	zeroGrad := tensor.NewTensor(make([]float32, numElements), inputShape)
	m.Children[0].Node.Backward(zeroGrad)
}

func (t *GraphTensor) ShapeOp(names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("shape_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewShapeOp(name, t)

	outputShape := []int{len(t.GetShape())}
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

func NewShapeOp(name string, a *GraphTensor) *ShapeOp {
	return &ShapeOp{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Shape",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
	}
}

func (m *ShapeOp) GetOutput() *tensor.Tensor {
	return m.output.value
}
