package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/Jimmy2099/torch/pkg/log"
)

func (l *BatchNormLayer) SetWeightsAndShape(data []float32, shape []int) {
	l.SetWeights(data)
}

func (l *BatchNormLayer) SetBiasAndShape(data []float32, shape []int) {
	l.SetBias(data)
}

type BatchNormLayer struct {
	weights     *tensor.Tensor
	bias        *tensor.Tensor
	RunningMean *tensor.Tensor
	runningVar  *tensor.Tensor
	eps         float32
	momentum    float32
	training    bool
	numFeatures int
}

func (l *BatchNormLayer) GetWeights() *tensor.Tensor {
	return l.weights
}

func (l *BatchNormLayer) GetBias() *tensor.Tensor {
	return l.bias
}

func NewBatchNormLayer(numFeatures int, eps, momentum float32) *BatchNormLayer {
	weights := tensor.Ones([]int{numFeatures})
	bias := tensor.Zeros([]int{numFeatures})
	runningMean := tensor.Zeros([]int{numFeatures})
	runningVar := tensor.Ones([]int{numFeatures})

	return &BatchNormLayer{
		weights:     weights,
		bias:        bias,
		RunningMean: runningMean,
		runningVar:  runningVar,
		eps:         eps,
		momentum:    momentum,
		numFeatures: numFeatures,
		training:    true,
	}
}

func (l *BatchNormLayer) SetWeights(data []float32) {
	if len(data) != l.numFeatures {
		panic(fmt.Sprintf("Weights data length mismatch: expected %d, got %d", l.numFeatures, len(data)))
	}
	l.weights = tensor.NewTensor(data, []int{l.numFeatures})
}

func (l *BatchNormLayer) SetBias(data []float32) {
	if len(data) != l.numFeatures {
		panic(fmt.Sprintf("bias data length mismatch: expected %d, got %d", l.numFeatures, len(data)))
	}
	l.bias = tensor.NewTensor(data, []int{l.numFeatures})
}

func (bn *BatchNormLayer) computeMean(x *tensor.Tensor) *tensor.Tensor {
	sumResult := x.SumByDim1([]int{0, 2, 3}, true)
	elementCount := float32(x.GetShape()[0] * x.GetShape()[2] * x.GetShape()[3])
	return sumResult.DivScalar(elementCount).Reshape([]int{bn.numFeatures})
}

func (bn *BatchNormLayer) computeVariance(x *tensor.Tensor, mean *tensor.Tensor) *tensor.Tensor {
	x_mu := x.Sub(mean.Reshape([]int{1, bn.numFeatures, 1, 1}))
	sq_diff := x_mu.Pow(2)
	sumResult := sq_diff.SumByDim1([]int{0, 2, 3}, true)
	elementCount := float32(x.GetShape()[0] * x.GetShape()[2] * x.GetShape()[3])

	return sumResult.DivScalar(elementCount).Reshape([]int{bn.numFeatures})
}

func (bn *BatchNormLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if bn.training {
		batchMean := bn.computeMean(x)
		batchVar := bn.computeVariance(x, batchMean)

		fmt.Printf("Training mode - batchMean shape: %v, batchVar shape: %v\n",
			batchMean.GetShape(), batchVar.GetShape())
		fmt.Printf("Running stats - mean shape: %v, var shape: %v\n",
			bn.RunningMean.GetShape(), bn.runningVar.GetShape())

		bn.updateRunningStats(batchMean, batchVar)

		broadcastDims := []int{1, bn.numFeatures, 1, 1}
		std := batchVar.AddScalar(bn.eps).Sqrt().Reshape(broadcastDims)
		return x.Sub(batchMean.Reshape(broadcastDims)).
			Div(std).
			Mul(bn.weights.Reshape(broadcastDims)).
			Add(bn.bias.Reshape(broadcastDims))
	} else {
		mean4d := bn.RunningMean.Reshape([]int{1, bn.numFeatures, 1, 1})
		std4d := bn.runningVar.AddScalar(bn.eps).Sqrt().Reshape([]int{1, bn.numFeatures, 1, 1})
		x_normalized := x.Sub(mean4d).Div(std4d)
		return x_normalized.Mul(bn.weights.Reshape([]int{1, bn.numFeatures, 1, 1})).Add(bn.bias.Reshape([]int{1, bn.numFeatures, 1, 1}))
	}
}

func (bn *BatchNormLayer) updateRunningStats(batchMean, batchVar *tensor.Tensor) {
	if !bn.RunningMean.ShapesMatch(batchMean) {
		log.Println(fmt.Sprintf("running mean shape mismatch: expect %v, got %v",
			bn.RunningMean.GetShape(), batchMean.GetShape()))
	}
	if !bn.runningVar.ShapesMatch(batchVar) {
		log.Println(fmt.Sprintf("running var shape mismatch: expect %v, got %v",
			bn.runningVar.GetShape(), batchVar.GetShape()))
	}
	newRunningMean := bn.RunningMean.MulScalar(1.0 - bn.momentum).Add(
		batchMean.MulScalar(bn.momentum),
	)
	newRunningVar := bn.runningVar.MulScalar(1.0 - bn.momentum).Add(
		batchVar.MulScalar(bn.momentum),
	)

	bn.RunningMean = newRunningMean
	bn.runningVar = newRunningVar
}
