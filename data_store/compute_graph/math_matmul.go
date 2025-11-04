package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type MatMul struct {
	*OPSNode
	OPSTensor
}

func (m *MatMul) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].Node.Forward()
	b := m.Children[1].Node.Forward()

	if len(a.GetShape()) < 2 || len(b.GetShape()) < 2 {
		panic("MatMul requires 2D tensors")
	}
	if a.GetShape()[len(a.GetShape())-1] != b.GetShape()[len(b.GetShape())-2] {
		panic(fmt.Sprintf("Incompatible dimensions for MatMul: %v and %v",
			a.GetShape(), b.GetShape()))
	}

	result := a.MatMul(b)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *MatMul) Backward(grad *tensor.Tensor) {
	aVal := m.Children[0].value
	bVal := m.Children[1].value

	if aVal == nil || bVal == nil || grad == nil {
		panic("nil tensor in matmul backward pass")
	}

	bTransposed := bVal.Transpose()
	gradA := grad.MatMul(bTransposed)

	aTransposed := aVal.Transpose()
	gradB := aTransposed.MatMul(grad)

	m.Children[0].Node.Backward(gradA)
	m.Children[1].Node.Backward(gradB)
}

func (t *GraphTensor) MatMul(other *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("matmul_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != other.Graph {
		panic("tensors belong to different graphs")
	}
	g := t.Graph

	if len(t.GetShape()) < 2 || len(other.GetShape()) < 2 {
		panic("MatMul requires tensors with at least 2 dimensions")
	}
	if t.GetShape()[len(t.GetShape())-1] != other.GetShape()[len(other.GetShape())-2] {
		panic(fmt.Sprintf("Incompatible dimensions for MatMul: %v and %v",
			t.GetShape(), other.GetShape()))
	}

	node := NewMatMul(name, t, other)

	outShape := make([]int, len(t.GetShape()))
	copy(outShape, t.GetShape())
	outShape[len(outShape)-1] = other.GetShape()[len(other.GetShape())-1]

	totalElements := 1
	for _, dim := range outShape {
		totalElements *= dim
	}

	valueData := make([]float32, totalElements)
	gradData := make([]float32, totalElements)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor(valueData, outShape),
		grad:  tensor.NewTensor(gradData, outShape),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape(outShape)

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewMatMul(name string, a, b *GraphTensor) *MatMul {
	return &MatMul{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "MatMul",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a, b},
		},
	}
}

func (m *MatMul) GetOutput() *tensor.Tensor {
	return m.output.value
}
