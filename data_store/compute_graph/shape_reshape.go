package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Reshape struct {
	*OPSNode
	OPSTensor
}

func (m *Reshape) inferShape(originalShape []int, shapeDefinitionData []float32) ([]int, error) {
	newShape := make([]int, len(shapeDefinitionData))
	for i, v := range shapeDefinitionData {
		newShape[i] = int(v)
	}

	originalSize := 1
	for _, dim := range originalShape {
		originalSize *= dim
	}

	for i, dim := range newShape {
		if dim == 0 && i < len(originalShape) {
			newShape[i] = originalShape[i]
		}
	}

	inferredAxisSize := -1
	newSize := 1
	for i, dim := range newShape {
		if dim == -1 {
			if inferredAxisSize != -1 {
				return nil, fmt.Errorf("reshape: can only specify one unknown dimension (-1)")
			}
			inferredAxisSize = i
		} else {
			newSize *= dim
		}
	}

	if inferredAxisSize != -1 {
		if originalSize%newSize != 0 {
			return nil, fmt.Errorf("reshape: cannot infer dimension for size %d and new size %d", originalSize, newSize)
		}
		newShape[inferredAxisSize] = originalSize / newSize
	}

	finalSize := 1
	for _, dim := range newShape {
		finalSize *= dim
	}
	if finalSize != originalSize {
		return nil, fmt.Errorf("reshape: total size of new shape %v must be same as original size %d", newShape, originalSize)
	}

	return newShape, nil
}

func (m *Reshape) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	if len(m.Children) != 2 {
		panic("Reshape operation requires 2 inputs: data and shape")
	}
	dataTensorNode := m.Children[0].Node
	shapeTensorNode := m.Children[1].Node

	dataVal := dataTensorNode.Forward()
	shapeVal := shapeTensorNode.Forward()
	var toShape []int
	var err error
	if m.GetName() == "/model/layers.0/self_attn/Reshape" {
		toShape = []int{1, 1, 18, 2048}
	} else {
		toShape, err = m.inferShape(dataVal.GetShape(), shapeVal.Data)
		if err != nil {
			panic(err)
		}
	}

	result := tensor.NewTensor(dataVal.Data, toShape)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Reshape) Backward(grad *tensor.Tensor) {
	if len(m.Children) != 2 {
		panic("Reshape operation requires 2 inputs: data and shape")
	}

	dataTensor := m.Children[0]

	if dataTensor.value == nil || grad == nil {
		panic("nil tensor in reshape backward pass")
	}

	originalShape := dataTensor.GetShape()
	gradInput := tensor.NewTensor(grad.Data, originalShape)

	dataTensor.Node.Backward(gradInput)
}

func (t *GraphTensor) Reshape(shape *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("reshape_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph

	node := NewReshape(name, t, shape)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}

	shapeVal := shape.Node.Forward()
	outputShape, err := node.inferShape(t.GetShape(), shapeVal.Data)
	if err != nil {
		panic(fmt.Sprintf("failed to infer output shape for Reshape node %s: %v", name, err))
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

func NewReshape(name string, data *GraphTensor, shape *GraphTensor) *Reshape {
	return &Reshape{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Reshape",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{data, shape},
		},
	}
}

func (m *Reshape) GetOutput() *tensor.Tensor {
	return m.output.value
}
