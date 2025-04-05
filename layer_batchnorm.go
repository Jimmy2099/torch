package torch

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/tensor" // 假设这个库存在
	"log"
	// "log" // 如果需要添加日志或调试信息
)

func (l *BatchNormLayer) SetWeightsAndShape(data []float64, shape []int) {
	l.SetWeights(data)
	// Ensure shape is compatible if needed, but typically keep (1,C,1,1)
	// l.gamma.Reshape(shape) // Be careful with this if shape is not (1,C,1,1)
}

func (l *BatchNormLayer) SetBiasAndShape(data []float64, shape []int) {
	l.SetBias(data)
	// Ensure shape is compatible if needed, but typically keep (1,C,1,1)
	// l.beta.Reshape(shape) // Be careful with this if shape is not (1,C,1,1)
}

type BatchNormLayer struct {
	weights     *tensor.Tensor
	bias        *tensor.Tensor
	RunningMean *tensor.Tensor
	runningVar  *tensor.Tensor
	eps         float64
	momentum    float64
	training    bool
	numFeatures int
}

func (l *BatchNormLayer) GetWeights() *tensor.Tensor {
	return l.weights
}

func (l *BatchNormLayer) GetBias() *tensor.Tensor {
	return l.bias
}

func NewBatchNormLayer(numFeatures int, eps, momentum float64) *BatchNormLayer {
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

func (l *BatchNormLayer) SetWeights(data []float64) {
	if len(data) != l.numFeatures {
		panic(fmt.Sprintf("Weights data length mismatch: expected %d, got %d", l.numFeatures, len(data)))
	}
	l.weights = tensor.NewTensor(data, []int{l.numFeatures})
}

func (l *BatchNormLayer) SetBias(data []float64) {
	if len(data) != l.numFeatures {
		panic(fmt.Sprintf("bias data length mismatch: expected %d, got %d", l.numFeatures, len(data)))
	}
	l.bias = tensor.NewTensor(data, []int{l.numFeatures})
}

func (bn *BatchNormLayer) computeMean(x *tensor.Tensor) *tensor.Tensor {
	sumResult := x.SumByDim1([]int{0, 2, 3}, true)
	elementCount := float64(x.Shape[0] * x.Shape[2] * x.Shape[3])
	return sumResult.DivScalar(elementCount).Reshape([]int{bn.numFeatures})
}

func (bn *BatchNormLayer) computeVariance(x *tensor.Tensor, mean *tensor.Tensor) *tensor.Tensor {
	x_mu := x.Sub(mean.Reshape([]int{1, bn.numFeatures, 1, 1}))
	sq_diff := x_mu.Pow(2)
	sumResult := sq_diff.SumByDim1([]int{0, 2, 3}, true)
	elementCount := float64(x.Shape[0] * x.Shape[2] * x.Shape[3])

	// 关键修复：使用有偏估计（PyTorch默认行为）
	return sumResult.DivScalar(elementCount).Reshape([]int{bn.numFeatures})
}

func (bn *BatchNormLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if bn.training {
		batchMean := bn.computeMean(x)               // 现在形状 [C]
		batchVar := bn.computeVariance(x, batchMean) // 现在形状 [C]

		// 添加调试输出
		fmt.Printf("Training mode - batchMean shape: %v, batchVar shape: %v\n",
			batchMean.Shape, batchVar.Shape)
		fmt.Printf("Running stats - mean shape: %v, var shape: %v\n",
			bn.RunningMean.Shape, bn.runningVar.Shape)

		bn.updateRunningStats(batchMean, batchVar)

		// 广播参数进行归一化
		broadcastDims := []int{1, bn.numFeatures, 1, 1}
		std := batchVar.AddScalar(bn.eps).Sqrt().Reshape(broadcastDims)
		return x.Sub(batchMean.Reshape(broadcastDims)).
			Div(std).
			Mul(bn.weights.Reshape(broadcastDims)).
			Add(bn.bias.Reshape(broadcastDims))
	} else {
		// 推理模式使用运行统计量
		mean4d := bn.RunningMean.Reshape([]int{1, bn.numFeatures, 1, 1})
		std4d := bn.runningVar.AddScalar(bn.eps).Sqrt().Reshape([]int{1, bn.numFeatures, 1, 1})
		x_normalized := x.Sub(mean4d).Div(std4d)
		return x_normalized.Mul(bn.weights.Reshape([]int{1, bn.numFeatures, 1, 1})).Add(bn.bias.Reshape([]int{1, bn.numFeatures, 1, 1}))
	}
}

func (bn *BatchNormLayer) updateRunningStats(batchMean, batchVar *tensor.Tensor) {
	// 增强的形状检查
	if !bn.RunningMean.ShapesMatch(batchMean) {
		log.Println(fmt.Sprintf("running mean shape mismatch: expect %v, got %v",
			bn.RunningMean.Shape, batchMean.Shape))
	}
	if !bn.runningVar.ShapesMatch(batchVar) {
		log.Println(fmt.Sprintf("running var shape mismatch: expect %v, got %v",
			bn.runningVar.Shape, batchVar.Shape))
	}
	// 关键修复：动量公式方向调整
	newRunningMean := bn.RunningMean.MulScalar(1.0 - bn.momentum).Add(
		batchMean.MulScalar(bn.momentum), // 新的权重是momentum
	)
	newRunningVar := bn.runningVar.MulScalar(1.0 - bn.momentum).Add(
		batchVar.MulScalar(bn.momentum), // 新的权重是momentum
	)

	bn.RunningMean = newRunningMean
	bn.runningVar = newRunningVar
}
