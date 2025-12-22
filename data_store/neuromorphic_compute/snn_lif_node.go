package neuromorphic_compute

import (
	"github.com/Jimmy2099/torch/data_store/compute_graph"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type LIFNode struct {
	Name      string
	Children  []*compute_graph.GraphTensor
	output    *compute_graph.GraphTensor
	Threshold float32
	Tau       float32
}

func NewLIFNode(name string, input *compute_graph.GraphTensor, threshold, tau float32) *LIFNode {
	return &LIFNode{
		Name:      name,
		Children:  []*compute_graph.GraphTensor{input},
		Threshold: threshold,
		Tau:       tau,
	}
}

func (n *LIFNode) Forward() *tensor.Tensor {
	if n.output.IsComputed() {
		return n.output.Value()
	}

	inputTensor := n.Children[0].Node.Forward()
	inputData := inputTensor.Data
	outputData := make([]float32, len(inputData))

	for i, val := range inputData {
		if val > n.Threshold {
			outputData[i] = 1.0
		} else {
			outputData[i] = 0.0
		}
	}

	outputShape := inputTensor.GetShape()
	outputTensor := tensor.NewTensor(outputData, outputShape)
	n.output.SetValue(outputTensor)
	n.output.SetComputed(true)
	return outputTensor
}

func (n *LIFNode) GetName() string { return n.Name }

func (n *LIFNode) ResetComputed() {
	n.output.SetComputed(false)
}

func (n *LIFNode) GetONNXNodeInfo() *compute_graph.ONNXNodeInfo {
	return &compute_graph.ONNXNodeInfo{
		Name:           "LIF",
		ProducedTensor: true,
	}
}

func (n *LIFNode) GetChildren() []compute_graph.Node {
	nodes := make([]compute_graph.Node, len(n.Children))
	for i, t := range n.Children {
		nodes[i] = t.Node
	}
	return nodes
}

func (n *LIFNode) GetOutput() *compute_graph.GraphTensor { return n.output }

func AddLIFNode(g *compute_graph.ComputationalGraph, input *compute_graph.GraphTensor, threshold, tau float32, name string) *compute_graph.GraphTensor {
	lifNode := NewLIFNode(name, input, threshold, tau)

	outputShape := input.Value().GetShape()
	outputTensor := &compute_graph.GraphTensor{
		Name:  name,
		Shape: outputShape,
		Graph: g,
		Node:  lifNode,
	}

	outputData := make([]float32, tensor.ShapeSum(outputShape))
	gradData := make([]float32, tensor.ShapeSum(outputShape))
	outputTensor.SetValue(tensor.NewTensor(outputData, outputShape))
	outputTensor.SetGrad(tensor.NewTensor(gradData, outputShape))

	if _, exists := g.Tensors[name]; exists {
		panic("Tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	lifNode.output = outputTensor
	g.Nodes = append(g.Nodes, lifNode)
	return outputTensor
}
