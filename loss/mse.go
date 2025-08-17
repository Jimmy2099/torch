package loss

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/compute_graph"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type MSELoss struct {
	Name   string
	Pred   *compute_graph.GraphTensor
	Target *compute_graph.GraphTensor
	output *compute_graph.GraphTensor

	*compute_graph.OPSNode
}

func NewMSELoss(name string, pred, target *compute_graph.GraphTensor) *MSELoss {
	return &MSELoss{
		Name:   name,
		Pred:   pred,
		Target: target,
	}
}

func MSE(g *compute_graph.ComputationalGraph, pred, target *compute_graph.GraphTensor, name string) *compute_graph.GraphTensor {
	if g != pred.Graph || g != target.Graph {
		panic("tensors belong to different graphs")
	}

	lossNode := NewMSELoss(name, pred, target)

	outputTensor := &compute_graph.GraphTensor{
		Name:  name,
		Shape: []int{1},
		Graph: g,
		Node:  lossNode,
	}
	outputTensor.SetValue(tensor.NewTensor([]float32{0}, []int{1}))
	outputTensor.SetGrad(tensor.NewTensor([]float32{0}, []int{1}))
	outputTensor.SetComputed(false)

	g.Tensors[name] = outputTensor
	lossNode.output = outputTensor
	g.Nodes = append(g.Nodes, lossNode)
	return outputTensor
}

func (m *MSELoss) Forward() *tensor.Tensor {
	if m.output.IsComputed() {
		return m.output.Value()
	}

	predVal := m.Pred.Node.Forward()
	targetVal := m.Target.Node.Forward()

	if len(predVal.Data) != len(targetVal.Data) {
		panic(fmt.Sprintf(
			"prediction (%d) and target (%d) sizes must match",
			len(predVal.Data),
			len(targetVal.Data),
		))
	}

	n := float32(len(predVal.Data))
	sumSquares := float32(0)
	for i := range predVal.Data {
		diff := predVal.Data[i] - targetVal.Data[i]
		sumSquares += diff * diff
	}

	loss := sumSquares / n
	result := tensor.NewTensor([]float32{loss}, []int{1})

	m.output.SetValue(result)
	m.output.SetComputed(true)
	return result
}

func (m *MSELoss) Backward(grad *tensor.Tensor) {
	if len(grad.Data) != 1 {
		panic("gradient for MSE loss must be scalar")
	}

	predVal := m.Pred.Value()
	targetVal := m.Target.Value()
	n := float32(len(predVal.Data))

	gradData := make([]float32, len(predVal.Data))
	for i := range gradData {
		gradData[i] = (2 / n) * (predVal.Data[i] - targetVal.Data[i]) * grad.Data[0]
	}

	gradTensor := tensor.NewTensor(gradData, predVal.GetShape())
	m.Pred.Node.Backward(gradTensor)
}

func (m *MSELoss) ResetComputed() {
	m.output.SetComputed(false)
}

func (m *MSELoss) GetName() string { return m.Name }

func (m *MSELoss) GetChildren() []compute_graph.Node {
	return []compute_graph.Node{m.Pred.Node, m.Target.Node}
}

func (m *MSELoss) GetOutput() *compute_graph.GraphTensor {
	return m.output
}
