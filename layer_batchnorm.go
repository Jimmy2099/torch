package torch

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/tensor" // 假设这个库存在
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
	runningMean *tensor.Tensor
	runningVar  *tensor.Tensor
	eps         float64
	momentum    float64
	training    bool
	numFeatures int
}

func NewBatchNormLayer(numFeatures int, eps, momentum float64) *BatchNormLayer {
	weights := tensor.Ones([]int{numFeatures})
	bias := tensor.Zeros([]int{numFeatures})
	runningMean := tensor.Zeros([]int{numFeatures})
	runningVar := tensor.Ones([]int{numFeatures})

	return &BatchNormLayer{
		weights:     weights,
		bias:        bias,
		runningMean: runningMean,
		runningVar:  runningVar,
		eps:         eps,
		momentum:    momentum,
		numFeatures: numFeatures,
		training:    true,
	}
}

func (l *BatchNormLayer) SetWeights(data []float64) {
	if len(data) != l.numFeatures {
		panic(fmt.Sprintf("weights data length mismatch: expected %d, got %d", l.numFeatures, len(data)))
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
	if len(x.Shape) != 4 || x.Shape[1] != bn.numFeatures {
		panic(fmt.Sprintf("input shape %v mismatch with num_features %d", x.Shape, bn.numFeatures))
	}

	sumResult := x.SumByDim1([]int{0, 2, 3}, true)
	elementCount := float64(x.Shape[0] * x.Shape[2] * x.Shape[3])
	mean := sumResult.DivScalar(elementCount)
	return mean.Reshape([]int{bn.numFeatures}) // 直接重塑为一维
}

func (bn *BatchNormLayer) computeVariance(x *tensor.Tensor, mean *tensor.Tensor) *tensor.Tensor {
	x_mu := x.Sub(mean.Reshape([]int{1, bn.numFeatures, 1, 1}))
	sq_diff := x_mu.Pow(2)
	sumResult := sq_diff.SumByDim1([]int{0, 2, 3}, true)
	elementCount := float64(x.Shape[0] * x.Shape[2] * x.Shape[3])
	variance := sumResult.DivScalar(elementCount)
	return variance.Reshape([]int{bn.numFeatures}) // 直接重塑为一维
}

func (bn *BatchNormLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if bn.training {
		batchMean := bn.computeMean(x)
		batchVar := bn.computeVariance(x, batchMean)
		bn.updateRunningStats(batchMean, batchVar)

		// 广播参数进行归一化
		mean4d := batchMean.Reshape([]int{1, bn.numFeatures, 1, 1})
		std4d := batchVar.AddScalar(bn.eps).Sqrt().Reshape([]int{1, bn.numFeatures, 1, 1})
		x_normalized := x.Sub(mean4d).Div(std4d)

		weights4d := bn.weights.Reshape([]int{1, bn.numFeatures, 1, 1})
		bias4d := bn.bias.Reshape([]int{1, bn.numFeatures, 1, 1})
		return x_normalized.Mul(weights4d).Add(bias4d)
	} else {
		// 推理模式使用运行统计量
		mean4d := bn.runningMean.Reshape([]int{1, bn.numFeatures, 1, 1})
		std4d := bn.runningVar.AddScalar(bn.eps).Sqrt().Reshape([]int{1, bn.numFeatures, 1, 1})
		x_normalized := x.Sub(mean4d).Div(std4d)
		return x_normalized.Mul(bn.weights.Reshape([]int{1, bn.numFeatures, 1, 1})).Add(bn.bias.Reshape([]int{1, bn.numFeatures, 1, 1}))
	}
}

func (bn *BatchNormLayer) updateRunningStats(batchMean, batchVar *tensor.Tensor) {
	// 形状一致性检查
	if !bn.runningMean.ShapesMatch(batchMean) || !bn.runningVar.ShapesMatch(batchVar) {
		panic(fmt.Sprintf("shape mismatch: running_mean%v vs batch_mean%v, running_var%v vs batch_var%v",
			bn.runningMean.Shape, batchMean.Shape, bn.runningVar.Shape, batchVar.Shape))
	}

	// 更新运行统计量
	newRunningMean := bn.runningMean.MulScalar(bn.momentum).Add(
		batchMean.MulScalar(1.0 - bn.momentum),
	)
	newRunningVar := bn.runningVar.MulScalar(bn.momentum).Add(
		batchVar.MulScalar(1.0 - bn.momentum),
	)

	bn.runningMean = newRunningMean
	bn.runningVar = newRunningVar
}
