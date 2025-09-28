package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Transpose struct {
	*OPSNode
	OPSTensor
	permutation        []int
	inversePermutation []int
}

func (m *Transpose) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	result := input.Permute(m.permutation)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Transpose) Backward(grad *tensor.Tensor) {
	transposedGrad := grad.Permute(m.inversePermutation)
	m.Children[0].Node.Backward(transposedGrad)
}

func (t *GraphTensor) Transpose(perm []int, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("transpose_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	inverse := make([]int, len(perm))
	for i, p := range perm {
		inverse[p] = i
	}

	g := t.Graph
	node := NewTranspose(name, t, perm, inverse)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
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

func NewTranspose(name string, a *GraphTensor, perm, inverse []int) *Transpose {
	return &Transpose{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Transpose",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a},
		},
		permutation:        perm,
		inversePermutation: inverse,
	}
}

func (m *Transpose) GetOutput() *GraphTensor {
	return m.output
}
