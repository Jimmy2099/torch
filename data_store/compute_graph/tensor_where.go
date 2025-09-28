package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Where struct {
	*OPSNode
	OPSTensor
}

func (m *Where) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	condition := m.Children[0].Node.Forward()
	trueValues := m.Children[1].Node.Forward()
	falseValues := m.Children[2].Node.Forward()

	if !tensor.ShapeEqual(condition.GetShape(), trueValues.GetShape()) ||
		!tensor.ShapeEqual(condition.GetShape(), falseValues.GetShape()) {
		panic("Where operation requires all inputs to have the same shape")
	}

	resultData := make([]float32, len(condition.Data))
	for i := range condition.Data {
		if condition.Data[i] != 0 {
			resultData[i] = trueValues.Data[i]
		} else {
			resultData[i] = falseValues.Data[i]
		}
	}

	result := tensor.NewTensor(resultData, condition.GetShape())
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Where) Backward(grad *tensor.Tensor) {
	condition := m.Children[0].Node.Forward()

	conditionGrad := tensor.NewTensor(make([]float32, len(condition.Data)), condition.GetShape())
	trueGrad := tensor.NewTensor(make([]float32, len(condition.Data)), condition.GetShape())
	falseGrad := tensor.NewTensor(make([]float32, len(condition.Data)), condition.GetShape())

	for i := range condition.Data {
		if condition.Data[i] != 0 {
			trueGrad.Data[i] = grad.Data[i]
			falseGrad.Data[i] = 0
		} else {
			trueGrad.Data[i] = 0
			falseGrad.Data[i] = grad.Data[i]
		}
		conditionGrad.Data[i] = 0
	}

	m.Children[0].Node.Backward(conditionGrad)
	m.Children[1].Node.Backward(trueGrad)
	m.Children[2].Node.Backward(falseGrad)
}

func (t *GraphTensor) Where(trueValues, falseValues *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("where_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewWhere(name, t, trueValues, falseValues)

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

func NewWhere(name string, condition, trueValues, falseValues *GraphTensor) *Where {
	return &Where{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Where",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{condition, trueValues, falseValues},
		},
	}
}

func (m *Where) GetOutput() *GraphTensor {
	return m.output
}
