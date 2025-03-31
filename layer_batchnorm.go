package torch

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/tensor" // 假设这个库存在
	// "log" // 如果需要添加日志或调试信息
)

func (l *BatchNormLayer) SetWeights(data []float64) {
	if len(data) != l.numFeatures {
		panic(fmt.Sprintf("weights (gamma) data length mismatch: expected %d, got %d", l.numFeatures, len(data)))
	}
	// Reshape to [1, C, 1, 1] for broadcasting
	l.gamma = tensor.NewTensor(data, []int{l.numFeatures})
}

func (l *BatchNormLayer) SetBias(data []float64) {
	if len(data) != l.numFeatures {
		panic(fmt.Sprintf("bias (beta) data length mismatch: expected %d, got %d", l.numFeatures, len(data)))
	}
	// Reshape to [1, C, 1, 1] for broadcasting
	l.beta = tensor.NewTensor(data, []int{l.numFeatures})
}

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

// BatchNormLayer 修改后的批量归一化层（无显式扩展）
type BatchNormLayer struct {
	gamma       *tensor.Tensor // 缩放参数 (Shape [C])
	beta        *tensor.Tensor // 平移参数 (Shape [C])
	runningMean *tensor.Tensor // 运行均值 (Shape [C])
	runningVar  *tensor.Tensor // 运行方差 (Shape [C])
	eps         float64
	momentum    float64
	training    bool
	numFeatures int
}

// NewBatchNormLayer 创建新层（参数形状改为 [C]）
func NewBatchNormLayer(numFeatures int, eps, momentum float64) *BatchNormLayer {
	gamma := tensor.Ones([]int{numFeatures}) // Shape [C]
	beta := tensor.Zeros([]int{numFeatures}) // Shape [C]
	runningMean := tensor.Zeros([]int{numFeatures})
	runningVar := tensor.Ones([]int{numFeatures})

	return &BatchNormLayer{
		gamma:       gamma,
		beta:        beta,
		runningMean: runningMean,
		runningVar:  runningVar,
		eps:         eps,
		momentum:    momentum,
		numFeatures: numFeatures,
		training:    true,
	}
}

func (bn *BatchNormLayer) computeMean(x *tensor.Tensor) *tensor.Tensor {
	// 输入形状验证
	if len(x.Shape) != 4 || x.Shape[1] != bn.numFeatures {
		panic(fmt.Sprintf("输入形状 %v 与特征数 %d 不匹配", x.Shape, bn.numFeatures))
	}

	// 计算求和（保持维度）
	sumResult := x.SumByDim1([]int{0, 2, 3}, true) // 结果形状 [1, C, 1, 1]

	// 打印中间结果验证
	fmt.Printf("Sum result shape: %v\n", sumResult.Shape)
	fmt.Printf("Sum values: %v\n", sumResult.Data[:3]) // 打印前3个通道的求和值

	// 计算平均值
	elementCount := float64(x.Shape[0] * x.Shape[2] * x.Shape[3])
	mean := sumResult.DivScalar(elementCount)

	// 移除冗余维度
	mean = mean.SqueezeSpecific([]int{0, 2, 3}) // 仅压缩指定维度

	// 打印最终结果
	fmt.Printf("Final mean shape: %v\n", mean.Shape)
	fmt.Printf("Mean values: %v\n", mean.Data)
	return mean
}

// computeVariance 同步修正
func (bn *BatchNormLayer) computeVariance(x *tensor.Tensor, mean *tensor.Tensor) *tensor.Tensor {
	// 输入形状 [N, C, H, W]
	x_mu := x.Sub(mean.Reshape([]int{1, bn.numFeatures, 1, 1}))
	sq_diff := x_mu.Pow(2)
	sumResult := sq_diff.SumByDim1([]int{0, 2, 3}, true) // [1, C, 1, 1]
	variance := sumResult.DivScalar(float64(x.Shape[0] * x.Shape[2] * x.Shape[3]))
	return variance.Squeeze() // [C]
}

// Forward 前向传播（无显式扩展）
func (bn *BatchNormLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if bn.training {
		mean := bn.computeMean(x)                                           // [C]
		variance := bn.computeVariance(x, mean)                             // [C]
		bn.updateRunningStats(mean, variance)                               // 更新统计量
		x_normalized := x.Sub(mean.Reshape([]int{1, bn.numFeatures, 1, 1})) // 广播减法
		x_normalized = x_normalized.Div(variance.AddScalar(bn.eps).Sqrt().Reshape([]int{1, bn.numFeatures, 1, 1}))
		return x_normalized.Mul(bn.gamma.Reshape([]int{1, bn.numFeatures, 1, 1})).Add(bn.beta.Reshape([]int{1, bn.numFeatures, 1, 1}))
	} else {
		x_normalized := x.Sub(bn.runningMean.Reshape([]int{1, bn.numFeatures, 1, 1}))
		x_normalized = x_normalized.Div(bn.runningVar.AddScalar(bn.eps).Sqrt().Reshape([]int{1, bn.numFeatures, 1, 1}))
		return x_normalized.Mul(bn.gamma.Reshape([]int{1, bn.numFeatures, 1, 1})).Add(bn.beta.Reshape([]int{1, bn.numFeatures, 1, 1}))
	}
}

// 在 BatchNormLayer 结构体定义后添加以下方法
func (bn *BatchNormLayer) updateRunningStats(batchMean, batchVar *tensor.Tensor) {
	// 确保形状匹配 [C]
	if !bn.runningMean.ShapesMatch(batchMean) || !bn.runningVar.ShapesMatch(batchVar) {
		panic("running stats shape mismatch with batch stats")
	}

	// 运行均值更新公式: running_mean = momentum * running_mean + (1 - momentum) * batch_mean
	newRunningMean := bn.runningMean.MulScalar(bn.momentum).Add(
		batchMean.MulScalar(1.0 - bn.momentum),
	)

	// 运行方差更新公式: running_var = momentum * running_var + (1 - momentum) * batch_var
	newRunningVar := bn.runningVar.MulScalar(bn.momentum).Add(
		batchVar.MulScalar(1.0 - bn.momentum),
	)

	// 更新字段（假设张量操作是不可变的，需要重新赋值）
	bn.runningMean = newRunningMean
	bn.runningVar = newRunningVar
}
