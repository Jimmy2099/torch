package compute_graph

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type NodeType int

const (
	InputNode NodeType = iota
	OutPutNode
	MultiplyNode
	AddNode
)

func (nt NodeType) String() string {
	switch nt {
	case InputNode:
		return "Input"
	case MultiplyNode:
		return "Multiply"
	case AddNode:
		return "Add"
	default:
		return "Unknown"
	}
}

func (n *Variable) GetOutput() *tensor.Tensor { return n.Output }
func (n *Variable) GetGrad() *tensor.Tensor   { return n.Grad }
func (n *Variable) GetName() string           { return n.Name }
func (n *Variable) GetChildren() []Node       { return n.Children }
func (n *Variable) AddChild(child Node)       { n.Children = append(n.Children, child) }
func (n *Variable) resetGrad()                { n.Grad = tensor.NewTensor([]float32{0}, []int{1}) }
