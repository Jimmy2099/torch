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
	if len(tensors) == 0 {
		return tensor.NewTensor([]float32{}, []int{})
	}

	outputShape := make([]int, len(tensors[0].GetShape()))
	copy(outputShape, tensors[0].GetShape())
	totalAxisSize := 0
	for _, t := range tensors {
		totalAxisSize += t.GetShape()[axis]
	}
	outputShape[axis] = totalAxisSize

	totalElements := 1
	for _, dim := range outputShape {
		totalElements *= dim
	}
	outputData := make([]float32, totalElements)

	outer_dims_size := 1
	for i := 0; i < axis; i++ {
		outer_dims_size *= outputShape[i]
	}

	inner_dims_size := 1
	for i := axis + 1; i < len(outputShape); i++ {
		inner_dims_size *= outputShape[i]
	}

	outputOffset := 0
	for i := 0; i < outer_dims_size; i++ {
		for _, t := range tensors {
			inputShape := t.GetShape()
			inputAxisDim := inputShape[axis]
			elementsToCopy := inputAxisDim * inner_dims_size
			inputOffset := i * elementsToCopy
			copy(outputData[outputOffset:], t.Data[inputOffset:inputOffset+elementsToCopy])
			outputOffset += elementsToCopy
		}
	}

	return tensor.NewTensor(outputData, outputShape)
}

func (m *Concat) splitGradient(grad *tensor.Tensor, axis int) []*tensor.Tensor {
	grads := make([]*tensor.Tensor, len(m.Children))
	if len(m.Children) == 0 {
		return grads
	}

	gradShape := grad.GetShape()
	outer_dims_size := 1
	for i := 0; i < axis; i++ {
		outer_dims_size *= gradShape[i]
	}

	inner_dims_size := 1
	for i := axis + 1; i < len(gradShape); i++ {
		inner_dims_size *= gradShape[i]
	}

	for i, child := range m.Children {
		childShape := child.Node.GetOutput().GetShape()
		childSize := 1
		for _, dim := range childShape {
			childSize *= dim
		}
		gradData := make([]float32, childSize)
		grads[i] = tensor.NewTensor(gradData, childShape)
	}

	gradOffset := 0
	for i := 0; i < outer_dims_size; i++ {
		for _, childGrad := range grads {
			childShape := childGrad.GetShape()
			childAxisDim := childShape[axis]
			elementsToCopy := childAxisDim * inner_dims_size
			childOffset := i * elementsToCopy
			copy(childGrad.Data[childOffset:], grad.Data[gradOffset:gradOffset+elementsToCopy])
			gradOffset += elementsToCopy
		}
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

	for i := 0; i < len(allInputs); i++ {
		currentShape := allInputs[i].GetShape()
		if len(currentShape) != len(inputShape) {
			panic("All inputs to Concat must have the same number of dimensions")
		}
		for j := 0; j < len(inputShape); j++ {
			if j != axis && inputShape[j] != currentShape[j] {
				panic(fmt.Sprintf("Dimension mismatch on non-axis dimension. Dim %d: %d vs %d", j, inputShape[j], currentShape[j]))
			}
		}
		if i > 0 {
			outputShape[axis] += currentShape[axis]
		}
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
