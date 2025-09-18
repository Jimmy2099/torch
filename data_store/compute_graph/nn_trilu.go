package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Trilu struct {
	*OPSNode
	OPSTensor
	upper bool
}

func (m *Trilu) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	kTensor := m.Children[1].Node.Forward()

	upper := true
	if len(m.Children) > 2 {
		upperTensor := m.Children[2].Node.Forward()
		if len(upperTensor.Data) > 0 {
			upper = upperTensor.Data[0] != 0
		}
	}

	k := 0
	if len(kTensor.Data) > 0 {
		k = int(kTensor.Data[0])
	}

	result := input.Trilu(k, upper)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Trilu) Backward(grad *tensor.Tensor) {

	inputShape := m.Children[0].Node.GetOutput().Shape

	input := m.Children[0].Node.Forward()
	kTensor := m.Children[1].Node.Forward()

	upper := true
	if len(m.Children) > 2 {
		upperTensor := m.Children[2].Node.Forward()
		if len(upperTensor.Data) > 0 {
			upper = upperTensor.Data[0] != 0
		}
	}

	k := 0
	if len(kTensor.Data) > 0 {
		k = int(kTensor.Data[0])
	}

	mask := input.TriluMask(k, upper)

	gradData := make([]float32, len(grad.Data))
	for i := range gradData {
		gradData[i] = grad.Data[i] * mask.Data[i]
	}

	gradInput := tensor.NewTensor(gradData, inputShape)
	m.Children[0].Node.Backward(gradInput)

	if len(m.Children) > 1 {
		zeroGrad := tensor.NewTensor(make([]float32, len(kTensor.Data)), kTensor.GetShape())
		m.Children[1].Node.Backward(zeroGrad)
	}
}

func (t *GraphTensor) Trilu(k *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("trilu_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewTrilu(name, t, k)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Shape: t.Shape,
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

func NewTrilu(name string, input, k *GraphTensor) *Trilu {
	return &Trilu{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Trilu",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{input, k},
		},
		upper: true,
	}
}

func (m *Trilu) GetOutput() *GraphTensor {
	return m.output
}
