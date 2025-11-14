package compute_graph

import (
	"errors"
	"github.com/Jimmy2099/torch/data_store/network"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

func OnnxNodeCompute(graph *ComputationalGraph, node *network.Node) ([]*tensor.Tensor, error) {
	onnx := NewOnnx()
	outputOnnx := onnx.NewOneTimeSessionTestByNode(graph, node)
	return outputOnnx, nil
}

func OutPutCompare(in1List []*tensor.Tensor, in2List []*tensor.Tensor) error {
	for _, in1 := range in1List {
		for _, in2 := range in2List {
			if !in1.Equal(in2) {
				return errors.New("outPutCompare contain different output")
			}
		}
	}
	return nil
}
