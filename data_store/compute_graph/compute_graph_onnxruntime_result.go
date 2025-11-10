package compute_graph

import (
	"errors"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

func ResultCompare(graph *ComputationalGraph) ([]*tensor.Tensor, []*tensor.Tensor, error) {
	var outputByGO []*tensor.Tensor
	{
		for _, j := range graph.Network.GetOutput() {
			outputByGO = append(outputByGO, graph.GetTensorByName(j.Name).Value())
		}
	}

	onnx := NewOnnx()
	outputOnnx := onnx.NewOneTimeSessionTest(graph)
	err := outPutCompare(outputByGO, outputOnnx)
	return outputByGO, outputOnnx, err
}

func outPutCompare(in1List []*tensor.Tensor, in2List []*tensor.Tensor) error {
	for _, in1 := range in1List {
		for _, in2 := range in2List {
			if !in1.Equal(in2) {
				return errors.New("outPutCompare contain different output")
			}
		}
	}
	return nil
}
