package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/node"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Concat struct {
	*OPSNode
	OPSTensor
	axis int
}

func (m *Concat) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	inputs := make([]*tensor.Tensor, len(m.Children))
	for i, child := range m.Children {
		inputs[i] = child.Node.Forward()
	}

	result := m.concatTensors(inputs, m.axis)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Concat) Backward(grad *tensor.Tensor) {
	grads := m.splitGradient(grad, m.axis)
	for i, child := range m.Children {
		child.Node.Backward(grads[i])
	}
}

func (m *Concat) concatTensors(tensors []*tensor.Tensor, axis int) *tensor.Tensor {
	totalSize := 0
	for _, t := range tensors {
		totalSize += t.GetShape()[axis]
	}

	outputShape := make([]int, len(tensors[0].GetShape()))
	copy(outputShape, tensors[0].GetShape())
	outputShape[axis] = totalSize

	// Calculate total elements manually
	totalElements := 1
	for _, dim := range outputShape {
		totalElements *= dim
	}

	outputData := make([]float32, totalElements)

	offset := 0
	for _, t := range tensors {
		copy(outputData[offset:offset+len(t.Data)], t.Data)
		offset += len(t.Data)
	}

	return tensor.NewTensor(outputData, outputShape)
}

func (m *Concat) splitGradient(grad *tensor.Tensor, axis int) []*tensor.Tensor {
	grads := make([]*tensor.Tensor, len(m.Children))

	splitPoints := make([]int, len(m.Children))
	for i, child := range m.Children {
		splitPoints[i] = child.Node.GetOutput().GetShape()[axis]
		if i > 0 {
			splitPoints[i] += splitPoints[i-1]
		}
	}

	offset := 0
	for i, child := range m.Children {
		// Remove unused size variable
		childOutput := child.Node.GetOutput()
		childTensor := childOutput
		childSize := len(childTensor.Data)

		gradData := make([]float32, childSize)
		copy(gradData, grad.Data[offset:offset+childSize])
		grads[i] = tensor.NewTensor(gradData, childOutput.GetShape())
		offset += childSize
	}

	return grads
}

func (t *GraphTensor) Concat(inputs []*GraphTensor, axis int, names ...string) *GraphTensor {
	if len(inputs) < 1 {
		panic("Concat requires at least 1 input tensor")
	}

	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("concat_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	allInputs := append([]*GraphTensor{t}, inputs...)

	g := t.Graph
	node := NewConcat(name, allInputs, axis)

	inputShape := t.GetShape()
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)

	for i := 1; i < len(allInputs); i++ {
		if len(allInputs[i].GetShape()) != len(inputShape) {
			panic("All inputs to Concat must have the same number of dimensions")
		}
		outputShape[axis] += allInputs[i].GetShape()[axis]
	}

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

func NewConcat(name string, inputs []*GraphTensor, axis int) *Concat {
	return &Concat{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Concat",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: inputs,
		},
		axis: axis,
	}
}

func (m *Concat) GetOutput() *tensor.Tensor {
	return m.output.value
}

func ConcatNodeRegistryFunc(name string, children []*GraphTensor, output *GraphTensor) node.Node {
	axis := 0
	m := NewConcat(name, children, axis)
	m.output = output
	return m
}
