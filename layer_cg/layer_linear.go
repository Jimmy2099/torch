package layer_cg

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/compute_graph"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type LinearLayer struct {
	num         int
	inFeatures  int
	outFeatures int
	weight      *compute_graph.GraphTensor
	bias        *compute_graph.GraphTensor
	weightGrad  *tensor.Tensor
	biasGrad    *tensor.Tensor
}

func NewLinearLayer(graph *compute_graph.ComputationalGraph, inFeatures, outFeatures int) *LinearLayer {
	weightData := make([]float32, inFeatures*outFeatures)
	biasData := make([]float32, outFeatures)

	num := graph.GetNameAutoInc("NewLinearLayer")

	weight := graph.NewGraphTensor(
		weightData,
		[]int{inFeatures, outFeatures},
		"linear_weight_"+fmt.Sprint(num),
	)

	bias := graph.NewGraphTensor(
		biasData,
		[]int{outFeatures},
		"linear_bias_"+fmt.Sprint(num),
	)

	return &LinearLayer{
		num:         num,
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
		weight:      weight,
		bias:        bias,
		weightGrad:  tensor.NewTensor(make([]float32, inFeatures*outFeatures), []int{inFeatures, outFeatures}),
		biasGrad:    tensor.NewTensor(make([]float32, outFeatures), []int{outFeatures}),
	}
}

func (l *LinearLayer) Forward(x *compute_graph.GraphTensor) *compute_graph.GraphTensor {

	matmul := x.MatMul(l.weight, "linear_matmul_"+fmt.Sprint(l.num))
	output := matmul.Add(l.bias, "linear_add_bias_"+fmt.Sprint(l.num))
	return output
}

func (l *LinearLayer) SetWeight(weight *compute_graph.GraphTensor) {
	l.weight = weight
}

func (l *LinearLayer) SetBias(bias *compute_graph.GraphTensor) {
	l.bias = bias
}
