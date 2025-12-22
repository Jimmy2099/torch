package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/node"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Add struct {
	*OPSNode
	Name     string
	Children []*GraphTensor
	output   *GraphTensor
}

func NewAdd(name string, a, b *GraphTensor) *Add {
	return &Add{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Add",
			ONNXProducedTensor: true,
		}),
		Name:     name,
		Children: []*GraphTensor{a, b},
	}
}

func (a *Add) Forward() *tensor.Tensor {
	if a.output.computed {
		return a.output.value
	}

	x := a.Children[0].Node.Forward()
	y := a.Children[1].Node.Forward()

	if len(x.Data) != len(y.Data) {
		panic("tensor sizes must match for addition")
	}

	result := x.Add(y)
	a.output.value = result
	a.output.computed = true
	return result
}

func (a *Add) ResetComputed() {
	a.output.computed = false
}

func (a *Add) GetName() string { return a.Name }

func (a *Add) GetChildren() []node.Node {
	nodes := make([]node.Node, len(a.Children))
	for i, t := range a.Children {
		nodes[i] = t.Node
	}
	return nodes
}

func (a *Add) GetOutput() *tensor.Tensor { return a.output.value }
