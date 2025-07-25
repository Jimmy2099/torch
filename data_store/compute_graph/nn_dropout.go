package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math/rand"
)

type Dropout struct {
	*OPSNode
	OPSTensor
	p    float32
	mask *tensor.Tensor
}

func (d *Dropout) Forward() *tensor.Tensor {
	if d.output.computed {
		return d.output.value
	}

	input := d.Children[0].Node.Forward()
	data := make([]float32, len(input.Data))
	d.mask = tensor.NewTensor(make([]float32, len(input.Data)), input.GetShape())

	scale := float32(1.0) / (1.0 - d.p)
	for i := range data {
		if rand.Float32() > d.p {
			data[i] = input.Data[i] * scale
			d.mask.Data[i] = 1.0
		}
	}

	result := tensor.NewTensor(data, input.GetShape())
	d.output.value = result
	d.output.computed = true
	return result
}

func (d *Dropout) Backward(grad *tensor.Tensor) {
	gradData := make([]float32, len(grad.Data))
	for i, v := range d.mask.Data {
		gradData[i] = grad.Data[i] * v
	}
	gradInput := tensor.NewTensor(gradData, grad.GetShape())
	d.Children[0].Node.Backward(gradInput)
}

func (t *GraphTensor) Dropout(p float32, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("dropout_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}
	g := t.Graph

	node := NewDropout(name, t, p)

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

func NewDropout(name string, input *GraphTensor, p float32) *Dropout {
	return &Dropout{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Dropout",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{input},
		},
		p: p,
	}
}

func (m *Dropout) GetOutput() *GraphTensor {
	return m.output
}
