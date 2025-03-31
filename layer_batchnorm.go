package torch

import (
	"fmt" // Import fmt for better error messages
	"github.com/Jimmy2099/torch/data_struct/tensor"
)

// BatchNormLayer 批量归一化层
type BatchNormLayer struct {
	gamma       *tensor.Tensor // 缩放参数 (Weight)
	beta        *tensor.Tensor // 平移参数 (Bias)
	runningMean *tensor.Tensor // 运行均值
	runningVar  *tensor.Tensor // 运行方差
	eps         float64        // 数值稳定项
	momentum    float64        // 动量系数
	training    bool           // 训练模式标志
	inputMean   *tensor.Tensor // 前向传播时计算的均值 (缓存)
	inputVar    *tensor.Tensor // 前向传播时计算的方差 (缓存)
	normalized  *tensor.Tensor // 归一化后的值 (缓存)
	input       *tensor.Tensor // 输入张量 (缓存，用于反向传播) - Added for potentially more accurate backward
	x_mu        *tensor.Tensor // x - mean (缓存，用于反向传播) - Added for potentially more accurate backward
	std_inv     *tensor.Tensor // 1 / sqrt(var + eps) (缓存，用于反向传播) - Added for potentially more accurate backward
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

	// 初始化参数 - Shape is [1, C, 1, 1] for easier broadcasting with (N, C, H, W)
	shape := []int{1, numFeatures, 1, 1} // Use broadcastable shape

	gamma := tensor.Ones(shape)
	beta := tensor.Zeros(shape)
	runningMean := tensor.Zeros(shape)
	runningVar := tensor.Ones(shape) // Initialize running variance to 1

	return &BatchNormLayer{
		gamma:       gamma,
		beta:        beta,
		runningMean: runningMean,
		runningVar:  runningVar,
		eps:         eps,
		momentum:    momentum,
		training:    true, // Default to training mode
		numFeatures: numFeatures,
	}
}

func (l *BatchNormLayer) SetWeights(data []float64) {
	if len(data) != l.numFeatures {
		panic(fmt.Sprintf("weights (gamma) data length mismatch: expected %d, got %d", l.numFeatures, len(data)))
	}
	// Reshape to [1, C, 1, 1] for broadcasting
	l.gamma = tensor.NewTensor(data, []int{1, l.numFeatures, 1, 1})
}

func (l *BatchNormLayer) SetBias(data []float64) {
	if len(data) != l.numFeatures {
		panic(fmt.Sprintf("bias (beta) data length mismatch: expected %d, got %d", l.numFeatures, len(data)))
	}
	// Reshape to [1, C, 1, 1] for broadcasting
	l.beta = tensor.NewTensor(data, []int{1, l.numFeatures, 1, 1})
}

func (l *BatchNormLayer) SetWeightsAndShape(data []float64, shape []int) {
	l.SetWeights(data)
	l.gamma.Reshape(shape)
}

func (l *BatchNormLayer) SetBiasAndShape(data []float64, shape []int) {
	l.SetBias(data)
	l.beta.Reshape(shape)
}

// normalize 归一化函数 (Internal helper, used in forward)
func (bn *BatchNormLayer) normalize(x, mean, variance *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	// x_mu = x - mean
	x_mu := x.Sub(mean) // Shape (N, C, H, W)
	// var_eps = variance + eps
	var_eps := variance.AddScalar(bn.eps) // Shape (1, C, 1, 1)
	// std_inv = 1 / sqrt(var_eps)
	std_inv := var_eps.Sqrt().Pow(-1) // Shape (1, C, 1, 1)
	// normalized = x_mu * std_inv
	normalized := x_mu.Mul(std_inv) // Shape (N, C, H, W)

	// Cache for backward pass
	bn.x_mu = x_mu
	bn.std_inv = std_inv
	bn.normalized = normalized

	return normalized, x_mu, std_inv // Return intermediate results if needed, mainly normalized
}

// scaleAndShift 缩放和平移 (Internal helper, used in forward)
func (bn *BatchNormLayer) scaleAndShift(normalized *tensor.Tensor) *tensor.Tensor {
	// y = normalized * gamma + beta
	return normalized.Mul(bn.gamma).Add(bn.beta) // Shapes (N, C, H, W), (1, C, 1, 1), (1, C, 1, 1) -> (N, C, H, W)
}

// computeMean 计算均值
func (bn *BatchNormLayer) computeMean(x *tensor.Tensor) *tensor.Tensor {
	// Input x shape: (N, C, H, W)
	// Mean calculation axis: (N, H, W) which are dims 0, 2, 3
	// Output mean shape should be (1, C, 1, 1) for broadcasting
	keepDims := true
	batchSize := float64(x.Shape[0] * x.Shape[2] * x.Shape[3])
	if batchSize <= 0 { // Avoid division by zero if H or W is 0? Should not happen ideally.
		batchSize = 1
	}
	mean := x.SumByDim1([]int{0, 2, 3}, keepDims).DivScalar(batchSize) // Corrected SumByDim usage
	bn.inputMean = mean                                                // Cache for backward (optional, as it can be recomputed)
	return mean
}

// computeVariance 计算方差
func (bn *BatchNormLayer) computeVariance(x *tensor.Tensor, mean *tensor.Tensor) *tensor.Tensor {
	// Input x shape: (N, C, H, W), mean shape: (1, C, 1, 1)
	// Variance calculation axis: (N, H, W) which are dims 0, 2, 3
	// Output variance shape should be (1, C, 1, 1) for broadcasting
	keepDims := true
	batchSize := float64(x.Shape[0] * x.Shape[2] * x.Shape[3])
	if batchSize <= 0 {
		batchSize = 1
	}
	// Variance = mean((x - mean)^2)
	sqDiff := x.Sub(mean).Pow(2)                                                // Shape (N, C, H, W)
	variance := sqDiff.SumByDim1([]int{0, 2, 3}, keepDims).DivScalar(batchSize) // Corrected SumByDim usage
	bn.inputVar = variance                                                      // Cache for backward
	return variance
}

// updateRunningStats 更新运行统计量
func (bn *BatchNormLayer) updateRunningStats(mean, variance *tensor.Tensor) {
	// running_mean = momentum * running_mean + (1 - momentum) * batch_mean
	bn.runningMean = bn.runningMean.MulScalar(bn.momentum).Add(
		mean.MulScalar(1.0 - bn.momentum),
	)

	// running_var = momentum * running_var + (1 - momentum) * batch_var
	// Note: PyTorch uses unbiased variance estimate for running_var update during training
	// batch_var_unbiased = batch_var * N / (N - 1) where N = batch_size * H * W
	// However, often the biased estimate is used for simplicity or if N is large.
	// Let's stick to the biased version first as per the original code's structure,
	// unless the tensor library makes unbiased easy.
	// If using unbiased:
	// N := float64(bn.input.Shape[0] * bn.input.Shape[2] * bn.input.Shape[3])
	// factor := N / (N - 1.0) // Bessel's correction only if N > 1
	// if N <= 1.0 { factor = 1.0 }
	// unbiasedVar := variance.MulScalar(factor)
	// bn.runningVar = bn.runningVar.MulScalar(bn.momentum).Add(
	// 	unbiasedVar.MulScalar(1.0 - bn.momentum),
	// )
	// Using biased variance for update:
	bn.runningVar = bn.runningVar.MulScalar(bn.momentum).Add(
		variance.MulScalar(1.0 - bn.momentum),
	)
}

// forwardTrain 训练模式前向传播
func (bn *BatchNormLayer) forwardTrain(x *tensor.Tensor) *tensor.Tensor {
	// Cache input for backward pass
	bn.input = x

	// 1. 计算批次均值和方差
	mean := bn.computeMean(x)               // Shape (1, C, 1, 1)
	variance := bn.computeVariance(x, mean) // Shape (1, C, 1, 1)

	// 2. 更新运行统计量 (using batch mean/variance)
	bn.updateRunningStats(mean, variance)

	// 3. 归一化 (using batch mean/variance)
	normalized, _, _ := bn.normalize(x, mean, variance) // Shapes (N, C, H, W)

	// 4. 缩放和平移 (using gamma/beta)
	output := bn.scaleAndShift(normalized) // Shape (N, C, H, W)

	return output
}

// forwardEval 推理模式前向传播
func (bn *BatchNormLayer) forwardEval(x *tensor.Tensor) *tensor.Tensor {
	// Use running mean and variance for normalization
	// Ensure runningMean/Var have the correct broadcastable shape (1, C, 1, 1)
	// No need to cache inputs or intermediate values for eval mode backward pass

	// x_mu = x - running_mean
	x_mu := x.Sub(bn.runningMean) // Shapes (N, C, H, W), (1, C, 1, 1) -> (N, C, H, W)
	// var_eps = running_var + eps
	var_eps := bn.runningVar.AddScalar(bn.eps) // Shape (1, C, 1, 1)
	// std_inv = 1 / sqrt(var_eps)
	std_inv := var_eps.Sqrt().Pow(-1) // Shape (1, C, 1, 1)
	// normalized = x_mu * std_inv
	normalized := x_mu.Mul(std_inv) // Shape (N, C, H, W)

	// Scale and shift
	output := normalized.Mul(bn.gamma).Add(bn.beta) // Shapes (N, C, H, W)
	return output
}

// Forward 前向传播
func (bn *BatchNormLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if x == nil {
		panic("input tensor cannot be nil")
	}

	// Assume input is (N, C, H, W) or (N, C)
	// We handle (N, C, H, W) here. Adapt if needed for (N, C).
	if len(x.Shape) < 2 {
		panic(fmt.Sprintf("input tensor must have at least 2 dimensions (N, C, ...), got %v", x.Shape))
	}
	if x.Shape[1] != bn.numFeatures {
		panic(fmt.Sprintf("input tensor channel dimension (%d) incompatible with batch norm layer's numFeatures (%d)", x.Shape[1], bn.numFeatures))
	}

	// Reshape parameters if input is 2D (N, C) -> (N, C, 1, 1) for consistency?
	// Or handle 2D separately. Assuming 4D (N, C, H, W) for now based on sum dims.

	if bn.training {
		return bn.forwardTrain(x)
	}
	return bn.forwardEval(x)
}

// Backward 反向传播
// Based on standard BatchNorm backward pass derivation
// https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
// https://arxiv.org/abs/1502.03167
func (bn *BatchNormLayer) Backward(dout *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	if !bn.training {
		panic("backward should only be called in training mode")
	}
	if bn.normalized == nil || bn.input == nil || bn.x_mu == nil || bn.std_inv == nil || bn.inputVar == nil {
		panic("backward called without forward pass or missing cached values")
	}
	if dout.Shape == nil || !dout.ShapesMatch(bn.input) {
		panic(fmt.Sprintf("dout shape %v mismatch with input shape %v", dout.Shape, bn.input.Shape))
	}

	// Input shapes:
	// dout:      (N, C, H, W) - Gradient from next layer
	// normalized:(N, C, H, W) - Cached from forward (x_hat)
	// gamma:     (1, C, 1, 1) - Layer parameter
	// std_inv:   (1, C, 1, 1) - Cached from forward (1 / sqrt(var + eps))
	// x_mu:      (N, C, H, W) - Cached from forward (x - mean)
	// inputVar:  (1, C, 1, 1) - Cached from forward (batch variance)

	N := float64(dout.Shape[0] * dout.Shape[2] * dout.Shape[3]) // Number of elements summed over per feature

	// 1. Compute gradient w.r.t. learnable parameters (gamma, beta)
	// dL/dbeta = sum(dL/dy) over N, H, W dims
	dbeta := dout.SumByDim1([]int{0, 2, 3}, true) // Keep shape (1, C, 1, 1)
	// dL/dgamma = sum(dL/dy * normalized) over N, H, W dims
	dgamma := dout.Mul(bn.normalized).SumByDim1([]int{0, 2, 3}, true) // Keep shape (1, C, 1, 1)

	// 2. Compute gradient w.r.t. normalized input (dxhat)
	// dL/dx_hat = dL/dy * gamma
	dxhat := dout.Mul(bn.gamma) // Shape (N, C, H, W)

	// 3. Compute gradient w.r.t. variance (dvar)
	// dL/dvar = sum(dL/dx_hat * (x - mu) * (-1/2) * (var + eps)^(-3/2)) over N, H, W
	// dvar = sum(dxhat * x_mu, dims) * (-0.5) * std_inv^3
	// Note: Need std_inv cubed, which is std_inv.Pow(3) or std_inv.Mul(std_inv).Mul(std_inv)
	dvar_term1 := dxhat.Mul(bn.x_mu)                       // Shape (N, C, H, W)
	dvar_sum := dvar_term1.SumByDim1([]int{0, 2, 3}, true) // Shape (1, C, 1, 1)
	// Efficiently calculate std_inv^3
	var_eps_sqrt := bn.inputVar.AddScalar(bn.eps).Sqrt() // sqrt(var+eps)
	var_eps_inv_3_2 := var_eps_sqrt.Pow(-3)              // (var+eps)^(-3/2) = std_inv^3
	// dvar := dvar_sum.Mul(std_inv.Pow(3)).MulScalar(-0.5) // Original thought, maybe less stable/efficient?
	dvar := dvar_sum.Mul(var_eps_inv_3_2).MulScalar(-0.5) // Shape (1, C, 1, 1)

	// 4. Compute gradient w.r.t. mean (dmu)
	// dL/dmu = sum(dL/dx_hat * (-1/std)) + dL/dvar * sum(-2 * (x - mu)) / N
	// dmu_term1 = sum(dxhat * (-std_inv)) over N, H, W
	dmu_term1_sum := dxhat.Mul(bn.std_inv.MulScalar(-1.0)).SumByDim1([]int{0, 2, 3}, true) // Shape (1, C, 1, 1)
	// dmu_term2 = dvar * sum(-2 * x_mu) / N = dvar * (-2/N) * sum(x_mu)
	dmu_term2_sum_xmu := bn.x_mu.SumByDim1([]int{0, 2, 3}, true) // Shape (1, C, 1, 1)
	dmu_term2 := dvar.Mul(dmu_term2_sum_xmu).MulScalar(-2.0 / N) // Shape (1, C, 1, 1)
	dmean := dmu_term1_sum.Add(dmu_term2)                        // Shape (1, C, 1, 1)

	// 5. Compute gradient w.r.t. input x (dx)
	// dL/dx = dL/dx_hat * (1/std) + dL/dvar * (2*(x-mu)/N) + dL/dmu * (1/N)
	// dx = dxhat * std_inv + dvar * (2 * x_mu / N) + dmean * (1 / N)
	dx_term1 := dxhat.Mul(bn.std_inv)                // Shape (N, C, H, W)
	dx_term2 := bn.x_mu.Mul(dvar).MulScalar(2.0 / N) // Shapes (N,C,H,W) * (1,C,1,1) * scalar -> (N,C,H,W)
	dx_term3 := dmean.MulScalar(1.0 / N)             // Shapes (1,C,1,1) * scalar -> (1,C,1,1), needs broadcasting

	dx := dx_term1.Add(dx_term2).Add(dx_term3) // Add broadcasts dmean appropriately

	// We don't update parameters here. Gradients are returned.
	// Optimizer will handle updates using these gradients.
	// bn.gamma = bn.gamma.Sub(dgamma.MulScalar(learning_rate)) // This belongs in optimizer step
	// bn.beta = bn.beta.Sub(dbeta.MulScalar(learning_rate))    // This belongs in optimizer step

	return dx, dgamma, dbeta // Return gradients: dx, dgamma, dbeta
}

// Parameters returns learnable parameters (gamma, beta)
func (bn *BatchNormLayer) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{bn.gamma, bn.beta}
}

// Grads should return the computed gradients after Backward() is called.
// Need to store dgamma and dbeta in the struct if this is the desired pattern.
// Let's modify Backward to return gradients instead.

// NumFeatures returns the number of features the layer operates on.
func (bn *BatchNormLayer) NumFeatures() int {
	return bn.numFeatures
}

// SetTraining sets the layer to training or evaluation mode.
func (bn *BatchNormLayer) SetTraining(training bool) {
	bn.training = training
}

// GetState returns the state (running mean, running var) for saving.
// Returns copies to prevent external modification.
func (l *BatchNormLayer) GetState() (runningMean []float64, runningVar []float64) {
	// Assuming Data() returns a flat []float64 representation
	meanCopy := make([]float64, len(l.runningMean.Data))
	copy(meanCopy, l.runningMean.Data)

	varCopy := make([]float64, len(l.runningVar.Data))
	copy(varCopy, l.runningVar.Data)

	return meanCopy, varCopy
}

// SetState sets the state (running mean, running var) for loading.
func (l *BatchNormLayer) SetState(runningMean []float64, runningVar []float64) {
	if len(runningMean) != l.numFeatures {
		panic(fmt.Sprintf("runningMean data length mismatch: expected %d, got %d", l.numFeatures, len(runningMean)))
	}
	if len(runningVar) != l.numFeatures {
		panic(fmt.Sprintf("runningVar data length mismatch: expected %d, got %d", l.numFeatures, len(runningVar)))
	}
	// Reshape to [1, C, 1, 1] for broadcasting
	l.runningMean = tensor.NewTensor(runningMean, []int{1, l.numFeatures, 1, 1})
	l.runningVar = tensor.NewTensor(runningVar, []int{1, l.numFeatures, 1, 1})
}
