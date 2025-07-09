package compute_graph

import (
	"encoding/json"
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"os"
)

type Node interface {
	Forward() *tensor.Tensor
	Backward(grad *tensor.Tensor)
	GetOutput() *tensor.Tensor
	GetGrad() *tensor.Tensor
	GetName() string
	GetChildren() []Node
}

type ComputationalGraph struct {
	nodes     []Node
	tensors   map[string]*Tensor
	output    *Tensor
	nodeCount int
}

func NewComputationalGraph() *ComputationalGraph {
	return &ComputationalGraph{
		tensors:   make(map[string]*Tensor),
		nodeCount: 0,
	}
}

func (g *ComputationalGraph) SetOutput(t *Tensor) {
	g.output = t
}

func (g *ComputationalGraph) GetOutput() *Tensor {
	return g.output
}

type Tensor struct {
	Name  string
	value *tensor.Tensor
	grad  *tensor.Tensor
	shape []int
	node  Node
	graph *ComputationalGraph
}

func (t *Tensor) Value() *tensor.Tensor {
	return t.value
}

func (t *Tensor) Grad() *tensor.Tensor {
	return t.grad
}

func (g *ComputationalGraph) NewTensor(data []float32, shape []int, name string) *Tensor {
	t := tensor.NewTensor(data, shape)
	if _, exists := g.tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}

	gradData := make([]float32, len(t.Data))
	grad := &tensor.Tensor{
		Data: gradData,
	}

	tensor := &Tensor{
		Name:  name,
		value: t,
		grad:  grad,
		shape: shape,
		graph: g,
	}

	g.tensors[name] = tensor

	node := &InputNode{
		Name:   "input:" + name,
		output: tensor,
	}
	tensor.node = node
	g.nodes = append(g.nodes, node)

	return tensor
}

func (t *Tensor) SetValue(value *tensor.Tensor) {
	t.value = value
}

type GraphExport struct {
	Nodes   []NodeExport   `json:"nodes"`
	Tensors []TensorExport `json:"tensors"`
	Output  string         `json:"output"`
}

type NodeExport struct {
	Name   string   `json:"name"`
	Type   string   `json:"type"`
	Inputs []string `json:"inputs"`
	Output string   `json:"output"`
}

type TensorExport struct {
	Name  string    `json:"name"`
	Value []float32 `json:"value"`
	Grad  []float32 `json:"grad"`
}

func LoadFromFile(filename string) (*ComputationalGraph, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	var exportData GraphExport
	if err := json.Unmarshal(data, &exportData); err != nil {
		return nil, fmt.Errorf("failed to unmarshal graph: %w", err)
	}

	graph := NewComputationalGraph()

	tensorMap := make(map[string]*Tensor)
	for _, te := range exportData.Tensors {
		t := &Tensor{
			Name:  te.Name,
			value: &tensor.Tensor{Data: te.Value},
			grad:  &tensor.Tensor{Data: te.Grad},
			graph: graph,
		}
		tensorMap[te.Name] = t
		graph.tensors[te.Name] = t
	}

	for _, ne := range exportData.Nodes {
		var node Node
		var inputs []*Tensor

		for _, inputName := range ne.Inputs {
			if inputTensor, ok := tensorMap[inputName]; ok {
				inputs = append(inputs, inputTensor)
			} else {
				return nil, fmt.Errorf("input tensor %s not found for node %s", inputName, ne.Name)
			}
		}

		outputTensor := tensorMap[ne.Output]
		if outputTensor == nil {
			return nil, fmt.Errorf("output tensor %s not found for node %s", ne.Output, ne.Name)
		}

		switch ne.Type {
		case "Input":
			node = &InputNode{
				Name:   ne.Name,
				output: outputTensor,
			}
		case "Multiply":
			if len(inputs) != 2 {
				return nil, fmt.Errorf("multiply node requires exactly 2 inputs")
			}
			node = &Multiply{
				Name:     ne.Name,
				Children: []*Tensor{inputs[0], inputs[1]},
				output:   outputTensor,
			}
		case "Add":
			if len(inputs) != 2 {
				return nil, fmt.Errorf("add node requires exactly 2 inputs")
			}
			node = &Add{
				Name:     ne.Name,
				Children: []*Tensor{inputs[0], inputs[1]},
				output:   outputTensor,
			}
		default:
			return nil, fmt.Errorf("unknown node type: %s", ne.Type)
		}

		outputTensor.node = node
		graph.nodes = append(graph.nodes, node)
	}

	if outputTensor, ok := tensorMap[exportData.Output]; ok {
		graph.output = outputTensor
	} else {
		return nil, fmt.Errorf("output tensor %s not found", exportData.Output)
	}

	return graph, nil
}

func (g *ComputationalGraph) GetTensors() map[string]*Tensor {
	return g.tensors
}

func (g *ComputationalGraph) Forward() {
	for _, node := range g.nodes {
		node.Forward()
	}
}

func (g *ComputationalGraph) Backward() {
	// Reset gradients
	for _, node := range g.nodes {
		if inputNode, ok := node.(*InputNode); ok {
			inputNode.resetGrad()
		}
	}

	// Perform backward pass starting from output
	for i := len(g.nodes) - 1; i >= 0; i-- {
		if _, ok := g.nodes[i].(*InputNode); !ok {
			g.nodes[i].Backward(tensor.NewTensor([]float32{1.0}, []int{1}))
			break
		}
	}
}

func (t *Tensor) Multiply(other *Tensor, names ...string) *Tensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("multiply_%d", t.graph.nodeCount)
		t.graph.nodeCount++
	}

	if t.graph != other.graph {
		panic("tensors belong to different graphs")
	}
	g := t.graph

	multNode := NewMultiply(name, t, other)

	outputTensor := &Tensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		shape: t.shape,
		graph: g,
		node:  multNode,
	}

	if _, exists := g.tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.tensors[name] = outputTensor
	multNode.output = outputTensor
	g.nodes = append(g.nodes, multNode)
	return outputTensor
}

func (t *Tensor) Add(other *Tensor, names ...string) *Tensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("add_%d", t.graph.nodeCount)
		t.graph.nodeCount++
	}

	if t.graph != other.graph {
		panic("tensors belong to different graphs")
	}
	g := t.graph

	addNode := NewAdd(name, t, other)

	outputTensor := &Tensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		shape: t.shape,
		graph: g,
		node:  addNode,
	}

	if _, exists := g.tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.tensors[name] = outputTensor
	addNode.output = outputTensor
	g.nodes = append(g.nodes, addNode)
	return outputTensor
}

// Input Node
type InputNode struct {
	Name   string
	Output *tensor.Tensor
	Grad   *tensor.Tensor
	output *Tensor
}

func (n *InputNode) Forward() *tensor.Tensor {
	n.Output = n.output.value
	return n.Output
}

func (n *InputNode) Backward(grad *tensor.Tensor) {
	if n.Grad == nil {
		n.Grad = tensor.NewTensor(make([]float32, len(n.Output.Data)), n.output.shape)
	}

	for i := range n.Grad.Data {
		n.Grad.Data[i] += grad.Data[i]
	}
}

func (n *InputNode) GetOutput() *tensor.Tensor { return n.Output }
func (n *InputNode) GetGrad() *tensor.Tensor   { return n.Grad }
func (n *InputNode) GetName() string           { return n.Name }
func (n *InputNode) GetChildren() []Node       { return nil }
func (n *InputNode) resetGrad() {
	n.Grad = tensor.NewTensor(make([]float32, len(n.Output.Data)), n.output.shape)
}

// Multiply Node
type Multiply struct {
	Name     string
	Children []*Tensor
	Output   *tensor.Tensor
	Grad     *tensor.Tensor
	output   *Tensor
}

func NewMultiply(name string, a, b *Tensor) *Multiply {
	return &Multiply{
		Name:     name,
		Children: []*Tensor{a, b},
	}
}

func (m *Multiply) Forward() *tensor.Tensor {
	a := m.Children[0].node.Forward()
	b := m.Children[1].node.Forward()

	if len(a.Data) != len(b.Data) {
		panic("tensor sizes must match for multiplication")
	}

	result := a.Mul(b)
	m.Output = result
	if m.output != nil {
		m.output.value = result
	}
	return result
}

func (m *Multiply) Backward(grad *tensor.Tensor) {
	m.Grad = grad
	a := m.Children[0].node.GetOutput()
	b := m.Children[1].node.GetOutput()

	// Calculate gradients for children
	gradA := b.Mul(grad)
	gradB := a.Mul(grad)

	// Propagate gradients
	m.Children[0].node.Backward(gradA)
	m.Children[1].node.Backward(gradB)
}

func (m *Multiply) GetOutput() *tensor.Tensor { return m.Output }
func (m *Multiply) GetGrad() *tensor.Tensor   { return m.Grad }
func (m *Multiply) GetName() string           { return m.Name }
func (m *Multiply) GetChildren() []Node {
	nodes := make([]Node, len(m.Children))
	for i, t := range m.Children {
		nodes[i] = t.node
	}
	return nodes
}

// Add Node
type Add struct {
	Name     string
	Children []*Tensor
	Output   *tensor.Tensor
	Grad     *tensor.Tensor
	output   *Tensor
}

func NewAdd(name string, a, b *Tensor) *Add {
	return &Add{
		Name:     name,
		Children: []*Tensor{a, b},
	}
}

func (a *Add) Forward() *tensor.Tensor {
	x := a.Children[0].node.Forward()
	y := a.Children[1].node.Forward()

	if len(x.Data) != len(y.Data) {
		panic("tensor sizes must match for addition")
	}

	result := x.Add(y)
	a.Output = result
	if a.output != nil {
		a.output.value = result
	}
	return result
}

func (a *Add) Backward(grad *tensor.Tensor) {
	a.Grad = grad
	// Propagate the same gradient to both children
	a.Children[0].node.Backward(grad)
	a.Children[1].node.Backward(grad)
}

func (a *Add) GetOutput() *tensor.Tensor { return a.Output }
func (a *Add) GetGrad() *tensor.Tensor   { return a.Grad }
func (a *Add) GetName() string           { return a.Name }
func (a *Add) GetChildren() []Node {
	nodes := make([]Node, len(a.Children))
	for i, t := range a.Children {
		nodes[i] = t.node
	}
	return nodes
}

func (g *ComputationalGraph) PrintStructure() {
	fmt.Println("\nComputation Graph Structure:")
	if g.output == nil {
		fmt.Println("  (No output node set)")
		return
	}

	// 从输出节点开始递归打印
	g.printNode(g.output.node, "", true, true)
}

func (g *ComputationalGraph) printNode(node Node, prefix string, isLast bool, isRoot bool) {
	// 确定当前节点的连接符号
	var connector string
	if isRoot {
		connector = "Output: "
	} else if isLast {
		connector = "└── "
	} else {
		connector = "├── "
	}

	// 打印当前节点
	fmt.Printf("%s%s%s (%s)\n", prefix, connector, node.GetName(), getNodeType(node))

	// 获取子节点
	children := node.GetChildren()
	if len(children) == 0 {
		return
	}

	// 更新前缀用于子节点
	newPrefix := prefix
	if isLast {
		newPrefix += "    "
	} else {
		newPrefix += "│   "
	}

	// 递归打印子节点
	for i, child := range children {
		isLastChild := i == len(children)-1
		g.printNode(child, newPrefix, isLastChild, false)
	}
}

func getNodeType(node Node) string {
	switch node.(type) {
	case *InputNode:
		return "Input"
	case *Multiply:
		return "Multiply"
	case *Add:
		return "Add"
	default:
		return "Operation"
	}
}
