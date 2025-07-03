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
	nodes   []Node
	tensors map[string]*Tensor
	output  *Tensor
}

func NewComputationalGraph() *ComputationalGraph {
	return &ComputationalGraph{
		tensors: make(map[string]*Tensor),
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
	node  *Variable
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
		graph: g,
	}

	g.tensors[name] = tensor

	node := &Variable{
		Name:   "input:" + name,
		OpType: InputNode,
		output: tensor,
	}
	tensor.node = node
	g.nodes = append(g.nodes, node)

	return tensor
}

//func (g *ComputationalGraph) Forward() {
//	for _, node := range g.nodes {
//		switch node.OpType {
//		case InputNode:
//		case MultiplyNode:
//			a := node.inputs[0].value
//			b := node.inputs[1].value
//			output := node.output.value
//
//			for i := range output.Data {
//				output.Data[i] = a.Data[i] * b.Data[i]
//			}
//
//		case AddNode:
//			a := node.inputs[0].value
//			b := node.inputs[1].value
//			output := node.output.value
//
//			for i := range output.Data {
//				output.Data[i] = a.Data[i] + b.Data[i]
//			}
//		}
//	}
//}

//func (g *ComputationalGraph) Backward() {
//	for _, t := range g.tensors {
//		for i := range t.grad.Data {
//			t.grad.Data[i] = 0
//		}
//	}
//
//	if g.output != nil {
//		for i := range g.output.grad.Data {
//			g.output.grad.Data[i] = 1.0
//		}
//	} else {
//		panic("output tensor not set")
//	}
//
//	for i := len(g.nodes) - 1; i >= 0; i-- {
//		node := g.nodes[i]
//		if node.gradFunc != nil {
//			node.gradFunc()
//		}
//	}
//}

func (t *Tensor) SetValue(value *tensor.Tensor) {
	t.value = value
}

//func (g *ComputationalGraph) PrintGraph() {
//	fmt.Println("Computational Graph Structure:")
//	for _, node := range g.nodes {
//		switch node.OpType {
//		case InputNode:
//			fmt.Printf("[Input] %s -> %s\n", node.Name, node.output.Name)
//		case MultiplyNode:
//			fmt.Printf("[Multiply] %s * %s -> %s\n",
//				node.inputs[0].Name, node.inputs[1].Name, node.output.Name)
//		case AddNode:
//			fmt.Printf("[Add] %s + %s -> %s\n",
//				node.inputs[0].Name, node.inputs[1].Name, node.output.Name)
//		}
//	}
//}

//func (g *ComputationalGraph) PrintDetailed() {
//	fmt.Println("Detailed Computational Graph:")
//	fmt.Println("Nodes:")
//	for _, node := range g.nodes {
//		fmt.Printf("  Node: %s (%s)\n", node.Name, node.OpType)
//
//		if len(node.inputs) > 0 {
//			fmt.Println("    Inputs:")
//			for _, input := range node.inputs {
//				fmt.Printf("      - %s\n", input.Name)
//			}
//		}
//
//		if node.output != nil {
//			fmt.Printf("    Output: %s\n", node.output.Name)
//		}
//	}
//
//	fmt.Println("\nTensors:")
//	for name, tensor := range g.tensors {
//		fmt.Printf("  Tensor: %s\n", name)
//		fmt.Printf("    Value: %v\n", tensor.value.Data)
//		fmt.Printf("    Grad: %v\n", tensor.grad.Data)
//	}
//
//	if g.output != nil {
//		fmt.Printf("\nOutput Tensor: %s\n", g.output.Name)
//	}
//}

type GraphExport struct {
	Nodes   []NodeExport   `json:"nodes"`
	Tensors []TensorExport `json:"tensors"`
	Output  string         `json:"output"`
}

type NodeExport struct {
	Name   string   `json:"name"`
	OpType NodeType `json:"opType"`
	Inputs []string `json:"inputs"`
	Output string   `json:"output"`
}

type TensorExport struct {
	Name  string    `json:"name"`
	Value []float32 `json:"value"`
	Grad  []float32 `json:"grad"`
}

//
//func (g *ComputationalGraph) ExportToFile(filename string) error {
//	exportData := GraphExport{
//		Output: g.output.Name,
//	}
//
//	for _, node := range g.nodes {
//		ne := NodeExport{
//			Name:   node.Name,
//			OpType: node.OpType,
//			Output: node.output.Name,
//		}
//
//		for _, input := range node.inputs {
//			ne.Inputs = append(ne.Inputs, input.Name)
//		}
//
//		exportData.Nodes = append(exportData.Nodes, ne)
//	}
//
//	for name, tensor := range g.tensors {
//		te := TensorExport{
//			Name:  name,
//			Value: tensor.value.Data,
//			Grad:  tensor.grad.Data,
//		}
//		exportData.Tensors = append(exportData.Tensors, te)
//	}
//
//	data, err := json.MarshalIndent(exportData, "", "  ")
//	if err != nil {
//		return fmt.Errorf("failed to marshal graph: %w", err)
//	}
//
//	err = os.WriteFile(filename, data, 0644)
//	if err != nil {
//		return fmt.Errorf("failed to write file: %w", err)
//	}
//
//	return nil
//}

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
			value: &tensor.Tensor{Data: te.Value}, // Fixed: create from exported Value
			grad:  &tensor.Tensor{Data: te.Grad},  // Already correct
			graph: graph,
		}

		tensorMap[te.Name] = t
		graph.tensors[te.Name] = t
	}

	for _, ne := range exportData.Nodes {
		node := &Variable{
			Name:   ne.Name,
			OpType: ne.OpType,
		}

		if outputTensor, ok := tensorMap[ne.Output]; ok {
			node.output = outputTensor
			outputTensor.node = node
		} else {
			return nil, fmt.Errorf("output tensor %s not found for node %s", ne.Output, ne.Name)
		}

		for _, inputName := range ne.Inputs {
			if inputTensor, ok := tensorMap[inputName]; ok {
				node.inputs = append(node.inputs, inputTensor)
			} else {
				return nil, fmt.Errorf("input tensor %s not found for node %s", inputName, ne.Name)
			}
		}

		switch ne.OpType {
		case MultiplyNode:
			node.gradFunc = func() {
				t := node.inputs[0]
				other := node.inputs[1]
				outputTensor := node.output
				for i := range t.grad.Data {
					t.grad.Data[i] += other.value.Data[i] * outputTensor.grad.Data[i]
				}
				for i := range other.grad.Data {
					other.grad.Data[i] += t.value.Data[i] * outputTensor.grad.Data[i]
				}
			}
		case AddNode:
			node.gradFunc = func() {
				t := node.inputs[0]
				other := node.inputs[1]
				outputTensor := node.output
				for i := range t.grad.Data {
					t.grad.Data[i] += outputTensor.grad.Data[i]
				}
				for i := range other.grad.Data {
					other.grad.Data[i] += outputTensor.grad.Data[i]
				}
			}
		}

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

//func (g *ComputationalGraph) GetNodes() []*Variable {
//	return g.nodes
//}

func (t *Tensor) Multiply(other *Tensor, name string) *Tensor {
	if t.graph != other.graph {
		panic("tensors belong to different graphs")
	}
	g := t.graph

	if len(t.value.Data) != len(other.value.Data) {
		panic("tensor sizes must match for multiplication")
	}

	outputValue := &tensor.Tensor{
		Data: make([]float32, len(t.value.Data)),
	}

	gradData := make([]float32, len(outputValue.Data))
	outputGrad := &tensor.Tensor{
		Data: gradData,
	}

	outputTensor := &Tensor{
		Name:  name,
		value: outputValue,
		grad:  outputGrad,
		graph: g,
	}

	if _, exists := g.tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.tensors[name] = outputTensor

	node := &Variable{
		Name:   name,
		OpType: MultiplyNode,
		inputs: []*Tensor{t, other},
		output: outputTensor,
		gradFunc: func() {
			for i := range t.grad.Data {
				t.grad.Data[i] += other.value.Data[i] * outputTensor.grad.Data[i]
			}

			for i := range other.grad.Data {
				other.grad.Data[i] += t.value.Data[i] * outputTensor.grad.Data[i]
			}
		},
	}
	outputTensor.node = node
	g.nodes = append(g.nodes, node)

	return outputTensor
}

func (t *Tensor) Add(other *Tensor, name string) *Tensor {
	if t.graph != other.graph {
		panic("tensors belong to different graphs")
	}
	g := t.graph

	if len(t.value.Data) != len(other.value.Data) {
		panic("tensor sizes must match for addition")
	}

	outputValue := &tensor.Tensor{
		Data: make([]float32, len(t.value.Data)),
	}

	gradData := make([]float32, len(outputValue.Data))
	outputGrad := &tensor.Tensor{
		Data: gradData,
	}

	outputTensor := &Tensor{
		Name:  name,
		value: outputValue,
		grad:  outputGrad,
		graph: g,
	}

	if _, exists := g.tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.tensors[name] = outputTensor

	node := &Variable{
		Name:   name,
		OpType: AddNode,
		inputs: []*Tensor{t, other},
		output: outputTensor,
		gradFunc: func() {
			for i := range t.grad.Data {
				t.grad.Data[i] += outputTensor.grad.Data[i]
			}

			for i := range other.grad.Data {
				other.grad.Data[i] += outputTensor.grad.Data[i]
			}
		},
	}
	outputTensor.node = node
	g.nodes = append(g.nodes, node)

	return outputTensor
}

//	func (g *ComputationalGraph) NewTensor(initValue *tensor.Tensor) *Variable {
//		val := newTensor(initValue)
//		g.AddNode(val)
//		val.graph = g
//		return val
//	}

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

//func (g *ComputationalGraph) AddNode(node Node) {
//	g.nodes = append(g.nodes, node)
//}

//func (g *ComputationalGraph) Print() {
//	fmt.Println("Computational Graph:")
//	for _, node := range g.nodes {
//		children := ""
//		for _, child := range node.GetChildren() {
//			children += child.GetName() + " "
//		}
//		if children == "" {
//			children = "None"
//		}
//		fmt.Printf("Node: %v | Output: %v | Grad: %v | Children: %s\n",
//			node.GetName(), node.GetOutput(), node.GetGrad(), children)
//	}
//}

func newTensor(initValue *tensor.Tensor) *Variable {
	return newVariable(initValue)
}
