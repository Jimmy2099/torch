package compute_graph

import (
	"fmt"
	"math"
	"sync"
)

type Node interface {
	Forward() float64
	Backward(grad float64)
	GetOutput() float64
	GetGrad() float64
	GetName() string
	GetChildren() []Node
}

type BaseNode struct {
	name     string
	output   float64
	grad     float64
	children []Node
	mu       sync.Mutex
}

func (n *BaseNode) GetOutput() float64  { return n.output }
func (n *BaseNode) GetGrad() float64    { return n.grad }
func (n *BaseNode) GetName() string     { return n.name }
func (n *BaseNode) GetChildren() []Node { return n.children }
func (n *BaseNode) addChild(child Node) { n.children = append(n.children, child) }
func (n *BaseNode) resetGrad()          { n.grad = 0 }

type Constant struct {
	BaseNode
	value float64
}

func NewConstant(name string, value float64) *Constant {
	return &Constant{
		BaseNode: BaseNode{name: name},
		value:    value,
	}
}

func (c *Constant) Forward() float64 {
	c.output = c.value
	return c.output
}

func (c *Constant) Backward(grad float64) {
}

type Variable struct {
	BaseNode
	value float64
}

func NewVariable(name string, initValue float64) *Variable {
	return &Variable{
		BaseNode: BaseNode{name: name},
		value:    initValue,
	}
}

func (v *Variable) Forward() float64 {
	v.output = v.value
	return v.output
}

func (v *Variable) Backward(grad float64) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.grad += grad
}

func (v *Variable) SetValue(value float64) {
	v.value = value
}

type Add struct {
	BaseNode
}

func NewAdd(name string, a, b Node) *Add {
	node := &Add{BaseNode: BaseNode{name: name}}
	node.addChild(a)
	node.addChild(b)
	return node
}

func (a *Add) Forward() float64 {
	a.output = a.children[0].Forward() + a.children[1].Forward()
	return a.output
}

func (a *Add) Backward(grad float64) {
	a.grad = grad
	a.children[0].Backward(grad)
	a.children[1].Backward(grad)
}

type Multiply struct {
	BaseNode
}

func NewMultiply(name string, a, b Node) *Multiply {
	node := &Multiply{BaseNode: BaseNode{name: name}}
	node.addChild(a)
	node.addChild(b)
	return node
}

func (m *Multiply) Forward() float64 {
	m.output = m.children[0].Forward() * m.children[1].Forward()
	return m.output
}

func (m *Multiply) Backward(grad float64) {
	m.grad = grad
	a := m.children[0].GetOutput()
	b := m.children[1].GetOutput()
	m.children[0].Backward(grad * b)
	m.children[1].Backward(grad * a)
}

type Sigmoid struct {
	BaseNode
}

func NewSigmoid(name string, input Node) *Sigmoid {
	node := &Sigmoid{BaseNode: BaseNode{name: name}}
	node.addChild(input)
	return node
}

func (s *Sigmoid) Forward() float64 {
	x := s.children[0].Forward()
	s.output = 1 / (1 + math.Exp(-x))
	return s.output
}

func (s *Sigmoid) Backward(grad float64) {
	s.grad = grad
	output := s.output
	localGrad := output * (1 - output)
	s.children[0].Backward(grad * localGrad)
}

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
			g.nodes[i].Backward(1.0)
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
		fmt.Printf("Node: %-10s | Output: %-10.4f | Grad: %-10.4f | Children: %s\n",
			node.GetName(), node.GetOutput(), node.GetGrad(), children)
	}
}
