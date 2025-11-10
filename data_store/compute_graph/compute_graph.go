package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/network"
	"github.com/Jimmy2099/torch/data_store/node"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"log"
	"os"
	"strings"
)

type ComputationalGraph struct {
	Nodes     []node.Node
	Tensors   map[string]*GraphTensor
	Network   *network.Network
	NodeCount int
	*ComputationalGraphCount
	*ONNXAttributePool
	*ComputationalGraphSettings
}

func NewComputationalGraph() *ComputationalGraph {
	var settings *ComputationalGraphSettings
	if os.Getenv("DebugMode") == "true" {
		settings = NewComputationalGraphSettings(ComputationalGraphRunModeDebug)
	} else {
		settings = NewComputationalGraphSettings(ComputationalGraphRunModeProduction)
	}

	return &ComputationalGraph{
		Tensors:                    make(map[string]*GraphTensor),
		NodeCount:                  0,
		ComputationalGraphCount:    NewComputationalGraphCount(),
		Network:                    network.NewNetwork(),
		ONNXAttributePool:          NewONNXAttributePool(),
		ComputationalGraphSettings: settings,
	}
}

func (g *ComputationalGraph) GetOutputNode() node.Node {
	return g.GetNodeByName(g.Network.GetOutput()[0].Name)
}

type GraphTensor struct {
	Name     string
	value    *tensor.Tensor
	grad     *tensor.Tensor
	shape    []int
	Node     node.Node
	Graph    *ComputationalGraph
	computed bool
}

func (t *GraphTensor) UpdateAll(graphTensor *GraphTensor) {
	if t == nil || graphTensor == nil {
		log.Println("UpdateAll Info: graphTensor is nil or graphTensor is nil")
		return
	}

	t.Name = graphTensor.Name
	t.value = graphTensor.value
	t.grad = graphTensor.grad
	t.shape = graphTensor.shape
	t.Node = graphTensor.Node
	t.Graph = graphTensor.Graph
	t.computed = graphTensor.computed
}

func (t *GraphTensor) GetShape() []int {
	if t.value.GetShape() == nil {
		return t.value.GetShape()
	}
	return t.shape
}

func (t *GraphTensor) SetShape(shape []int) {
	t.shape = shape
}

func (t *GraphTensor) Value() *tensor.Tensor {
	return t.value
}

func (t *GraphTensor) Grad() *tensor.Tensor {
	return t.grad
}

func (t *GraphTensor) IsComputed() bool {
	return t.computed
}

func (t *GraphTensor) SetComputed(computed bool) {
	t.computed = computed
}

func (t *GraphTensor) SetValue(value *tensor.Tensor) {
	t.value = value
}

func (t *GraphTensor) SetGrad(grad *tensor.Tensor) {
	t.grad = grad
}

func (g *ComputationalGraph) SetInput(t *GraphTensor, data []float32) {
	t.value.Data = data
}

func (g *ComputationalGraph) NewGraphTensor(data []float32, shape []int, name string) *GraphTensor {
	t := tensor.NewTensor(data, shape)
	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}

	gradData := make([]float32, len(t.Data))
	grad := tensor.NewTensor(gradData, shape)

	te := &GraphTensor{
		Name:     name,
		value:    t,
		grad:     grad,
		shape:    shape,
		Graph:    g,
		computed: false,
	}

	g.Tensors[name] = te

	node := &InputNode{
		Name:   name,
		output: te,
	}
	te.Node = node
	g.Nodes = append(g.Nodes, node)

	return te
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

func (g *ComputationalGraph) GetTensors() map[string]*GraphTensor {
	return g.Tensors
}

func (g *ComputationalGraph) GetTensorByName(name string) *GraphTensor {
	gNode := g.Network.GetNodeByName(name)
	if gNode == nil {
		return nil
	}
	for k, v := range g.Tensors {
		if k == name {
			return v
		}
	}
	t := &GraphTensor{
		Name: name,
	}
	g.Tensors[name] = t
	return t
}

// Forward TODO output-driven  multi-output node
// Forward TODO impl input-driven for bio neuro-fire and make bio-neuro like chip
func (g *ComputationalGraph) Forward() {
	g.Reset()

	if len(g.Network.GetOutput()) == 0 {
		panic("Error: Computational graph output is not set.")
	}
	visited := make(map[*network.Node]bool)

	for _, output := range g.Network.GetOutput() {
		g.forwardNode(output, visited)
	}
}

func (g *ComputationalGraph) forwardNode(n *network.Node, visited map[*network.Node]bool) {
	if n == nil {
		return
	}
	if visited[n] {
		return
	}
	visited[n] = true

	if strings.LastIndex(n.Type, "Tensor_") == 0 || len(n.Outputs) == 0 {
		if n.Parent != nil {
			g.forwardNode(n.Parent, visited)
		} else {
		}
		return
	}

	for _, in := range n.Inputs {
		if in == nil {
			continue
		}
		g.forwardNode(in, visited)
	}

	graphNode := g.GetNodeByName(n.Name)
	if graphNode == nil {
		panic("graphNode is null: " + n.Name)
	}

	if g.IsDebugMode() {
		graphNode.Forward()
		r1Go, r2Onnx, err := ResultCompareByNode(g, g.Network.GetNodeByName(graphNode.GetName()))
		if err == nil {
			return
		}
		_, _, _ = r1Go, r2Onnx, err
		log.Println("ResultCompareByNode error:", err)
		log.Println("set result by onnxruntime compute result!!!")
		t := g.GetTensorByName(g.Network.GetNodeByName(graphNode.GetName()).GetOutputName()[0])
		t.value = r2Onnx[0]
	} else {
		graphNode.Forward()
	}
}

func (g *ComputationalGraph) GetOutput() *GraphTensor {
	nodeList := g.Network.GetOutput()
	//TODO
	return g.GetTensorByName(nodeList[0].Name)
}

func (g *ComputationalGraph) Backward() {
	for _, node := range g.Nodes {
		if inputNode, ok := node.(*InputNode); ok {
			inputNode.resetGrad()
		}
	}

	if len(g.Network.GetOutput()) == 0 {
		return
	}

	outputShape := g.GetOutput().value.GetShape()
	num := 1
	for _, dim := range outputShape {
		num *= dim
	}
	gradData := make([]float32, num)
	for i := range gradData {
		gradData[i] = 1.0
	}
	gradTensor := tensor.NewTensor(gradData, outputShape)

	g.GetOutput().Node.Backward(gradTensor)
}

func (t *GraphTensor) Multiply(other *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("multiply_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != other.Graph {
		panic("tensors belong to different graphs")
	}
	g := t.Graph

	multNode := NewMultiply(name, t, other)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  multNode,
	}
	outputTensor.SetShape(t.GetShape())

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	multNode.output = outputTensor
	g.Nodes = append(g.Nodes, multNode)
	return outputTensor
}

func (t *GraphTensor) Add(other *GraphTensor, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("add_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != other.Graph {
		panic("tensors belong to different graphs")
	}
	g := t.Graph

	addNode := NewAdd(name, t, other)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  addNode,
	}
	outputTensor.SetShape(t.GetShape())

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	addNode.output = outputTensor
	g.Nodes = append(g.Nodes, addNode)
	return outputTensor
}

func (g *ComputationalGraph) Reset() {
	for _, tensor := range g.Tensors {
		if tensor != nil {
			tensor.SetComputed(false)
		}
	}
}

// PrintSmallModelStructure debug purpose
func (g *ComputationalGraph) PrintSmallModelStructure() {
	fmt.Println("\nComputation Graph Structure:")
	if g.GetOutput() == nil {
		fmt.Println("  (No output node set)")
		return
	}
	//Todo
	g.debugPrintNode(g.Network.GetOutput()[0], "", true, true)
}

// debugPrintNode debug purpose
func (g *ComputationalGraph) debugPrintNode(node *network.Node, prefix string, isLast bool, isRoot bool) {
	var connector string
	if isRoot {
		connector = "Output: "
	} else if isLast {
		connector = "└── "
	} else {
		connector = "├── "
	}

	if node == nil {
		node = &network.Node{Name: "Input_Nil"}
	}

	fmt.Printf("%s%s%s (%s)\n", prefix, connector, node.Name, node.Type)

	children := node.Inputs
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
		g.debugPrintNode(child, newPrefix, isLastChild, false)
	}
}

func (g *ComputationalGraph) GetNodeByName(name string) node.Node {
	for i := 0; i < len(g.Nodes); i++ {
		if g.Nodes[i].GetName() == name {
			return g.Nodes[i]
		}
	}

	ne := g.Network.GetNodeByName(name)
	if ne == nil {
		panic(fmt.Sprintf("node '%s' not found in network definition", name))
	}

	var onnxNodeInfo *ONNXOperator
	{
		if onnxNodeInfo = GetONNXNodeInfoByName(ne.Type); onnxNodeInfo == nil {
			panic(fmt.Sprintf("unknown node type: %s", ne.Type))
		} else if 1 == 2 {
			if len(ne.Outputs) != 1 { //TODO
				panic(fmt.Sprintf("invalid output length: %d", len(ne.Outputs)))
			}
			if onnxNodeInfo.InputPCount != -1 {
				if len(ne.Inputs) != onnxNodeInfo.InputPCount && len(ne.Outputs) != onnxNodeInfo.OutputPCount {
					panic(fmt.Sprintf("node %s of type %s has %d inputs but %d inputs were expected, got %d , %d",
						ne.Name, ne.Type, onnxNodeInfo.InputPCount, onnxNodeInfo.OutputPCount, len(ne.Inputs), 1))
				}

			}
		}
	}

	var inputs []*GraphTensor
	for _, inputName := range ne.Inputs {
		inputGraphTensor := g.GetTensorByName(inputName.Name)
		if inputGraphTensor == nil {
			panic("input tensor not found: " + inputName.Name)
		}

		if inputGraphTensor.Node == nil {
			newNode := &InputNode{
				Name:   inputGraphTensor.Name,
				output: inputGraphTensor,
			}
			inputGraphTensor.Node = newNode
			g.Nodes = append(g.Nodes, newNode)
		}

		inputs = append(inputs, inputGraphTensor)
	}

	var outputs []*GraphTensor
	for _, outputName := range ne.Outputs {
		outputGraphTensor := g.GetTensorByName(outputName.Name)
		if outputGraphTensor == nil {
			panic("output tensor could not be null: " + outputName.Name)
		}
		outputs = append(outputs, outputGraphTensor)
	}

	var output *GraphTensor
	if len(outputs) > 0 {
		output = outputs[0]
	}
	gNode := onnxNodeInfo.NodeRegistryFunc(ne.Name, inputs, output) //TODO outputs[0]
	if gNode == nil {
		panic(fmt.Sprintf("failed to create node %s of type %s", ne.Name, ne.Type))
	}

	if output != nil {
		output.Node = gNode
	}

	g.Nodes = append(g.Nodes, gNode)
	return gNode
}
