package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	math "github.com/chewxy/math32"
)

func (t *GraphTensor) Pow(exponent *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("pow_tensor_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != exponent.Graph {
		panic("tensors belong to different graphs")
	}

	node := NewPow(name, t, exponent)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Shape: t.Shape,
		Graph: t.Graph,
		Node:  node,
	}

	t.Graph.Tensors[name] = outputTensor
	node.output = outputTensor
	t.Graph.Nodes = append(t.Graph.Nodes, node)
	return outputTensor
}

type Pow struct {
	*OPSNode
	OPSTensor
}

func NewPow(name string, base, exponent *GraphTensor) *Pow {
	return &Pow{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Pow",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{base, exponent},
		},
	}
}

func (m *Pow) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	base := m.Children[0].Node.Forward()
	exponent := m.Children[1].Node.Forward()

	if len(base.Data) != len(exponent.Data) {
		panic("tensor sizes must match for power operation")
	}

	result := make([]float32, len(base.Data))
	//TODO broadcast
	for i := range base.Data {
		result[i] = math.Pow(base.Data[i], exponent.Data[i])
	}
	m.output.value = tensor.NewTensor(result, base.GetShape())
	m.output.computed = true
	return m.output.value
}

func (m *Pow) Backward(grad *tensor.Tensor) {
	baseVal := m.Children[0].value
	exponentVal := m.Children[1].value

	if baseVal == nil || exponentVal == nil || grad == nil {
		panic("nil tensor in power backward pass")
	}

	gradBase := make([]float32, len(baseVal.Data))
	for i := range baseVal.Data {
		gradBase[i] = grad.Data[i] * exponentVal.Data[i] *
			math.Pow(baseVal.Data[i], exponentVal.Data[i]-1)
	}

	gradExp := make([]float32, len(exponentVal.Data))
	for i := range exponentVal.Data {
		if baseVal.Data[i] <= 0 {
			panic("cannot compute gradient for exponent when base <= 0")
		}
		gradExp[i] = grad.Data[i] *
			(math.Pow(baseVal.Data[i], exponentVal.Data[i])) *
			(math.Log(baseVal.Data[i]))
	}

	m.Children[0].Node.Backward(tensor.NewTensor(gradBase, baseVal.GetShape()))
	m.Children[1].Node.Backward(tensor.NewTensor(gradExp, exponentVal.GetShape()))
}

func (m *Pow) GetOutput() *GraphTensor {
	return m.output
}
