package torch

import (
	"github.com/Jimmy2099/torch/data_struct/tensor"
)

// BatchNormLayer 批量归一化层
type BatchNormLayer struct {
	gamma       *tensor.Tensor // 缩放参数
	beta        *tensor.Tensor // 平移参数
	runningMean *tensor.Tensor // 运行均值
	runningVar  *tensor.Tensor // 运行方差
	eps         float64        // 数值稳定项
	momentum    float64        // 动量系数
	training    bool           // 训练模式标志
	inputMean   *tensor.Tensor // 前向传播时计算的均值
	inputVar    *tensor.Tensor // 前向传播时计算的方差
	normalized  *tensor.Tensor // 归一化后的值
	numFeatures int            // 特征维度大小
}

// NewBatchNormLayer 创建新的批量归一化层
func NewBatchNormLayer(numFeatures int, eps float64, momentum float64) *BatchNormLayer {
	if numFeatures <= 0 {
		panic("numFeatures must be positive")
	}
	if eps <= 0 {
		panic("eps must be positive")
	}
	if momentum < 0 || momentum > 1 {
		panic("momentum must be between 0 and 1")
	}

	// 初始化参数
	gamma := tensor.Ones([]int{numFeatures})
	beta := tensor.Zeros([]int{numFeatures})
	runningMean := tensor.Zeros([]int{numFeatures})
	runningVar := tensor.Ones([]int{numFeatures})

	return &BatchNormLayer{
		gamma:       gamma,
		beta:        beta,
		runningMean: runningMean,
		runningVar:  runningVar,
		eps:         eps,
		momentum:    momentum,
		training:    true,
		numFeatures: numFeatures,
	}
}

// normalize 归一化函数
func (bn *BatchNormLayer) normalize(x, mean, variance *tensor.Tensor) *tensor.Tensor {
	// 实现归一化逻辑
	// 需要确保Tensor有Sub、Div、AddScalar、Sqrt等方法
	return x.Sub(mean).Div(variance.AddScalar(bn.eps)).Sqrt()
}

// forwardTrain 训练模式前向传播
func (bn *BatchNormLayer) forwardTrain(x *tensor.Tensor) *tensor.Tensor {
	// 1. 计算均值和方差
	mean := bn.computeMean(x)
	variance := bn.computeVariance(x, mean) // 将var改为variance

	// 2. 更新运行统计量
	bn.updateRunningStats(mean, variance) // 将var改为variance

	// 3. 归一化
	bn.normalized = bn.normalize(x, mean, variance) // 将var改为variance

	// 4. 缩放和平移
	return bn.scaleAndShift(bn.normalized)
}

// computeVariance 计算方差
func (bn *BatchNormLayer) computeVariance(x *tensor.Tensor, mean *tensor.Tensor) *tensor.Tensor {
	// 沿(N,H,W)维度求方差，保留通道维度
	sqDiff := x.Sub(mean).Pow(2)
	variance := sqDiff.SumByDim1([]int{0, 2, 3}, true).DivScalar( // 将var改为variance
		float64(x.Shape[0] * x.Shape[2] * x.Shape[3]),
	)
	bn.inputVar = variance // 将var改为variance
	return variance        // 将var改为variance
}

// updateRunningStats 更新运行统计量
func (bn *BatchNormLayer) updateRunningStats(mean, variance *tensor.Tensor) { // 将var改为variance
	// 更新运行均值
	bn.runningMean = bn.runningMean.MulScalar(bn.momentum).Add(
		mean.MulScalar(1 - bn.momentum),
	)

	// 更新运行方差 (使用无偏估计)
	unbiasedVar := variance.MulScalar( // 将var改为variance
		float64(bn.numFeatures) / float64(bn.numFeatures-1),
	)
	bn.runningVar = bn.runningVar.MulScalar(bn.momentum).Add(
		unbiasedVar.MulScalar(1 - bn.momentum),
	)
}

// Forward 前向传播
func (bn *BatchNormLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if x == nil {
		panic("input tensor cannot be nil")
	}

	// 检查输入维度 (N,C,...)
	if len(x.Shape) < 2 || x.Shape[1] != bn.numFeatures {
		panic("input tensor shape incompatible with batch norm layer")
	}

	if bn.training {
		// 训练模式
		return bn.forwardTrain(x)
	}
	// 推理模式
	return bn.forwardEval(x)
}

// forwardEval 推理模式前向传播
func (bn *BatchNormLayer) forwardEval(x *tensor.Tensor) *tensor.Tensor {
	// 使用运行统计量归一化
	normalized := x.Sub(bn.runningMean).Div(
		bn.runningVar.AddScalar(bn.eps).Sqrt(),
	)
	return normalized.Mul(bn.gamma).Add(bn.beta)
}

// Backward 反向传播
func (bn *BatchNormLayer) Backward(dout *tensor.Tensor) *tensor.Tensor {
	if !bn.training {
		panic("backward should only be called in training mode")
	}
	if bn.normalized == nil {
		panic("must call forward first")
	}

	// 1. 计算dbeta和dgamma
	N := float64(dout.Shape[0] * dout.Shape[2] * dout.Shape[3])
	dgamma := bn.normalized.Mul(dout).SumByDim1([]int{0, 2, 3}, true)
	dbeta := dout.SumByDim1([]int{0, 2, 3}, true)

	// 2. 计算dxhat
	dxhat := dout.Mul(bn.gamma)

	// 3. 计算dvar
	dvar := dxhat.Mul(bn.normalized).SumByDim1([]int{0}, true).
		MulScalar(-0.5).Mul(
		bn.inputVar.AddScalar(bn.eps).Pow(-1.5),
	)

	// 4. 计算dmean
	dmean := dxhat.Mul(
		bn.inputVar.AddScalar(bn.eps).Sqrt().MulScalar(-1),
	).SumByDim1([]int{0}, true)
	dmean = dmean.Add(
		bn.normalized.Mul(dvar).MulScalar(-2.0/N).SumByDim1([]int{0}, true),
	)

	// 5. 计算dx
	dx := dxhat.Div(bn.inputVar.AddScalar(bn.eps).Sqrt())
	dx = dx.Add(bn.normalized.Mul(dvar).MulScalar(2.0 / N))
	dx = dx.Add(dmean.MulScalar(1.0 / N))

	// 更新参数梯度
	bn.gamma = bn.gamma.Sub(dgamma.MulScalar(bn.momentum))
	bn.beta = bn.beta.Sub(dbeta.MulScalar(bn.momentum))

	return dx
}

// computeMean 计算均值
func (bn *BatchNormLayer) computeMean(x *tensor.Tensor) *tensor.Tensor {
	// 沿(N,H,W)维度求平均，保留通道维度
	mean := x.SumByDim1([]int{0, 2, 3}, true).DivScalar(
		float64(x.Shape[0] * x.Shape[2] * x.Shape[3]),
	)
	bn.inputMean = mean
	return mean
}

// scaleAndShift 缩放和平移
func (bn *BatchNormLayer) scaleAndShift(normalized *tensor.Tensor) *tensor.Tensor {
	return normalized.Mul(bn.gamma).Add(bn.beta)
}

// SetTraining 设置训练/推理模式
func (bn *BatchNormLayer) SetTraining(training bool) {
	bn.training = training
}

// Parameters 返回可训练参数
func (bn *BatchNormLayer) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{bn.gamma, bn.beta}
}

// NumFeatures 返回特征维度
func (bn *BatchNormLayer) NumFeatures() int {
	return bn.numFeatures
}
