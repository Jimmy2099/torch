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
	GetName() string
	GetChildren() []Node
	ResetComputed()
}

type ComputationalGraph struct {
	nodes     []Node
	tensors   map[string]*GraphTensor
	output    *GraphTensor
	nodeCount int
}

func NewComputationalGraph() *ComputationalGraph {
	return &ComputationalGraph{
		tensors:   make(map[string]*GraphTensor),
		nodeCount: 0,
	}
}

func (g *ComputationalGraph) SetOutput(t *GraphTensor) {
	g.output = t
}

func (g *ComputationalGraph) GetOutput() *GraphTensor {
	return g.output
}

type GraphTensor struct {
	Name     string
	value    *tensor.Tensor
	grad     *tensor.Tensor
	shape    []int
	node     Node
	graph    *ComputationalGraph
	computed bool
}

func (t *GraphTensor) Value() *tensor.Tensor {
	return t.value
}

func (t *GraphTensor) Grad() *tensor.Tensor {
	return t.grad
}

func (g *ComputationalGraph) NewGraphTensor(data []float32, shape []int, name string) *GraphTensor {
	t := tensor.NewTensor(data, shape)
	if _, exists := g.tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}

	gradData := make([]float32, len(t.Data))
	grad := tensor.NewTensor(gradData, shape)

	tensor := &GraphTensor{
		Name:     name,
		value:    t,
		grad:     grad,
		shape:    shape,
		graph:    g,
		computed: false,
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

func (t *GraphTensor) SetValue(value *tensor.Tensor) {
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

	tensorMap := make(map[string]*GraphTensor)
	for _, te := range exportData.Tensors {
		t := &GraphTensor{
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
		var inputs []*GraphTensor

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
				Children: []*GraphTensor{inputs[0], inputs[1]},
				output:   outputTensor,
			}
		case "Add":
			if len(inputs) != 2 {
				return nil, fmt.Errorf("add node requires exactly 2 inputs")
			}
			node = &Add{
				Name:     ne.Name,
				Children: []*GraphTensor{inputs[0], inputs[1]},
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

func (g *ComputationalGraph) GetTensors() map[string]*GraphTensor {
	return g.tensors
}

func (g *ComputationalGraph) Forward() {
	for _, node := range g.nodes {
		node.ResetComputed()
	}

	if g.output != nil {
		g.output.node.Forward()
	}
}

func (g *ComputationalGraph) Backward() {
	for _, node := range g.nodes {
		if inputNode, ok := node.(*InputNode); ok {
			inputNode.resetGrad()
		}
	}

	if g.output == nil {
		return
	}

	outputShape := g.output.value.GetShape()
	num := 1
	for _, dim := range outputShape {
		num *= dim
	}
	gradData := make([]float32, num)
	for i := range gradData {
		gradData[i] = 1.0
	}
	gradTensor := tensor.NewTensor(gradData, outputShape)

	g.output.node.Backward(gradTensor)
}

func (t *GraphTensor) Multiply(other *GraphTensor, names ...string) *GraphTensor {
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

	outputTensor := &GraphTensor{
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

func (t *GraphTensor) Add(other *GraphTensor, names ...string) *GraphTensor {
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

	outputTensor := &GraphTensor{
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

type InputNode struct {
	Name   string
	Output *tensor.Tensor
	Grad   *tensor.Tensor
	output *GraphTensor
}

func (n *InputNode) Forward() *tensor.Tensor {
	if n.output.computed {
		return n.output.value
	}
	n.output.computed = true
	return n.output.value
}

func (n *InputNode) Backward(grad *tensor.Tensor) {
	if n.output.grad == nil {
		n.output.grad = tensor.NewTensor(
			make([]float32, len(n.output.value.Data)),
			n.output.value.GetShape(),
		)
	}

	for i := range grad.Data {
		n.output.grad.Data[i] += grad.Data[i]
	}
}

func (n *InputNode) resetGrad() {
	n.output.grad = tensor.NewTensor(make([]float32, len(n.output.value.Data)), n.output.value.GetShape())
}

func (n *InputNode) ResetComputed() {
	n.output.computed = false
}

func (n *InputNode) GetOutput() *tensor.Tensor { return n.output.value }
func (n *InputNode) GetGrad() *tensor.Tensor   { return n.Grad }
func (n *InputNode) GetName() string           { return n.Name }
func (n *InputNode) GetChildren() []Node       { return nil }

type Multiply struct {
	Name     string
	Children []*GraphTensor
	output   *GraphTensor
}

func NewMultiply(name string, a, b *GraphTensor) *Multiply {
	return &Multiply{
		Name:     name,
		Children: []*GraphTensor{a, b},
	}
}

func (m *Multiply) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	a := m.Children[0].node.Forward()
	b := m.Children[1].node.Forward()

	if len(a.Data) != len(b.Data) {
		panic("tensor sizes must match for multiplication")
	}

	result := a.Mul(b)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Multiply) ResetComputed() {
	m.output.computed = false
}

func (m *Multiply) Backward(grad *tensor.Tensor) {
	aVal := m.Children[0].value
	bVal := m.Children[1].value

	if aVal == nil || bVal == nil || grad == nil {
		panic("nil tensor in Multiply backward pass")
	}

	gradA := bVal.Mul(grad)
	gradB := aVal.Mul(grad)

	m.Children[0].node.Backward(gradA)
	m.Children[1].node.Backward(gradB)
}

func (m *Multiply) GetName() string { return m.Name }
func (m *Multiply) GetChildren() []Node {
	nodes := make([]Node, len(m.Children))
	for i, t := range m.Children {
		nodes[i] = t.node
	}
	return nodes
}
func (m *Multiply) GetOutput() *tensor.Tensor { return m.output.value }

func (a *Add) GetOutput() *tensor.Tensor { return a.output.value }

type Add struct {
	Name     string
	Children []*GraphTensor
	output   *GraphTensor
}

func NewAdd(name string, a, b *GraphTensor) *Add {
	return &Add{
		Name:     name,
		Children: []*GraphTensor{a, b},
	}
}

func (a *Add) Forward() *tensor.Tensor {
	if a.output.computed {
		return a.output.value
	}

	x := a.Children[0].node.Forward()
	y := a.Children[1].node.Forward()

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

func (a *Add) Backward(grad *tensor.Tensor) {
	a.Children[0].node.Backward(grad)
	a.Children[1].node.Backward(grad)
}

func (a *Add) GetName() string { return a.Name }

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

	g.printNode(g.output.node, "", true, true)
}

func (g *ComputationalGraph) printNode(node Node, prefix string, isLast bool, isRoot bool) {
	var connector string
	if isRoot {
		connector = "Output: "
	} else if isLast {
		connector = "└── "
	} else {
		connector = "├── "
	}

	fmt.Printf("%s%s%s (%s)\n", prefix, connector, node.GetName(), getNodeType(node))

	children := node.GetChildren()
	if len(children) == 0 {
		return
	}

	newPrefix := prefix
	if isLast {
		newPrefix += "    "
	} else {
		newPrefix += "│   "
	}

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
