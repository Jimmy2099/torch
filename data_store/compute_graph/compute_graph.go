package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"sync"
)

type Node interface {
	Forward() *tensor.Tensor
	Backward(grad *tensor.Tensor)
	GetOutput() *tensor.Tensor
	GetGrad() *tensor.Tensor
	GetName() string
	GetChildren() []Node
}

type BaseNode struct {
	Name     string
	Output   *tensor.Tensor
	Grad     *tensor.Tensor
	Children []Node
	mu       sync.Mutex
}

func (n *BaseNode) GetOutput() *tensor.Tensor { return n.Output }
func (n *BaseNode) GetGrad() *tensor.Tensor   { return n.Grad }
func (n *BaseNode) GetName() string           { return n.Name }
func (n *BaseNode) GetChildren() []Node       { return n.Children }
func (n *BaseNode) AddChild(child Node)       { n.Children = append(n.Children, child) }
func (n *BaseNode) resetGrad()                { n.Grad = tensor.NewTensor([]float32{0}, []int{1}) }

type ComputationalGraph struct {
	nodes []Node
}

func NewComputationalGraph() *ComputationalGraph {
	return &ComputationalGraph{}
}

func (g *ComputationalGraph) AddNode(node Node) {
	g.nodes = append(g.nodes, node)
}

func (g *ComputationalGraph) Forward() {
	for _, node := range g.nodes {
		node.Forward()
	}
}

func (g *ComputationalGraph) Backward() {
	for i := len(g.nodes) - 1; i >= 0; i-- {
		if v, ok := g.nodes[i].(*Variable); ok {
			v.resetGrad()
		}
	}

	for i := len(g.nodes) - 1; i >= 0; i-- {
		if _, ok := g.nodes[i].(*Variable); !ok {
			g.nodes[i].Backward(tensor.NewTensor([]float32{1.0}, []int{1}))
			break
		}
	}
}

func (g *ComputationalGraph) Print() {
	fmt.Println("Computational Graph:")
	for _, node := range g.nodes {
		children := ""
		for _, child := range node.GetChildren() {
			children += child.GetName() + " "
		}
		if children == "" {
			children = "None"
		}
		fmt.Printf("Node: %v | Output: %v | Grad: %v | Children: %s\n",
			node.GetName(), node.GetOutput(), node.GetGrad(), children)
	}
}
