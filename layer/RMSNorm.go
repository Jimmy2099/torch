package layer

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	math "github.com/chewxy/math32"
	"runtime"
	"sync"
)

// RMSNorm implements Root Mean Square Layer Normalization.
type RMSNorm struct {
	Weights    *tensor.Tensor // Scale parameters (gamma)
	eps        float32        // Small constant for numerical stability
	inputCache *tensor.Tensor // Cache for input tensor needed in backward pass
}

func (r *RMSNorm) GetWeights() *tensor.Tensor {
	return r.Weights
}

func (r *RMSNorm) GetBias() *tensor.Tensor {
	return nil // RMSNorm typically doesn't use bias
}

// NewRMSNorm creates a new RMSNorm layer.
func NewRMSNorm(features int, eps float32) *RMSNorm {
	// Initialize weights (gamma) with ones
	weightsData := make([]float32, features)
	for i := range weightsData {
		weightsData[i] = 1.0
	}

	weights := tensor.NewTensor(weightsData, []int{features})

	return &RMSNorm{
		Weights:    weights,
		eps:        eps,
		inputCache: nil,
	}
}

// Forward performs the RMSNorm forward pass.
func (r *RMSNorm) ForwardSignalThread(inputTensor *tensor.Tensor) *tensor.Tensor {
	if len(inputTensor.Shape) == 1 {
		inputTensor = inputTensor.Reshape([]int{1, inputTensor.Shape[0]})
	}
	// Input validation
	if len(inputTensor.Shape) < 2 {
		panic(fmt.Sprintf("Input must be at least 2D tensor, got shape: %v", inputTensor.Shape))
	}

	features := inputTensor.Shape[len(inputTensor.Shape)-1]
	if features != r.Weights.Shape[0] {
		panic(fmt.Sprintf("Feature dimension mismatch: input has %d, weights have %d", features, r.Weights.Shape[0]))
	}

	// Cache input for backward pass
	r.inputCache = inputTensor

	// Calculate output shape (same as input)
	outputShape := inputTensor.Shape
	outputData := make([]float32, len(inputTensor.Data))
	output := tensor.NewTensor(outputData, outputShape)

	// Calculate mean square for each feature vector
	batchSize := product(inputTensor.Shape[:len(inputTensor.Shape)-1])
	featureSize := features

	for b := 0; b < batchSize; b++ {
		start := b * featureSize
		end := start + featureSize

		// Calculate mean square
		sumSq := float32(0.0)
		for i := start; i < end; i++ {
			sumSq += inputTensor.Data[i] * inputTensor.Data[i]
		}
		meanSq := sumSq / float32(featureSize)

		// Calculate RMS
		rms := math.Sqrt(meanSq + r.eps)

		// Normalize and scale
		for i := start; i < end; i++ {
			featureIdx := i % featureSize
			output.Data[i] = (inputTensor.Data[i] / rms) * r.Weights.Data[featureIdx]
		}
	}

	return output
}

var numCPU = runtime.NumCPU()

func (r *RMSNorm) Forward(inputTensor *tensor.Tensor) *tensor.Tensor {
	if len(inputTensor.Shape) == 1 {
		inputTensor = inputTensor.Reshape([]int{1, inputTensor.Shape[0]})
	}

	// 输入验证保持不变
	if len(inputTensor.Shape) < 2 {
		panic(fmt.Sprintf("Input must be at least 2D tensor, got shape: %v", inputTensor.Shape))
	}

	features := inputTensor.Shape[len(inputTensor.Shape)-1]
	if features != r.Weights.Shape[0] {
		panic(fmt.Sprintf("Feature dimension mismatch: input has %d, weights have %d", features, r.Weights.Shape[0]))
	}

	r.inputCache = inputTensor

	outputShape := inputTensor.Shape
	outputData := make([]float32, len(inputTensor.Data))
	output := tensor.NewTensor(outputData, outputShape)

	batchSize := product(inputTensor.Shape[:len(inputTensor.Shape)-1])
	featureSize := features

	// 并发控制参数

	chunkSize := (batchSize + numCPU - 1) / numCPU // 计算每个分块的大小
	var wg sync.WaitGroup

	// 使用带缓冲的channel控制并发数
	sem := make(chan struct{}, numCPU*2) // 2倍核心数的缓冲区

	for chunkStart := 0; chunkStart < batchSize; chunkStart += chunkSize {
		wg.Add(1)
		sem <- struct{}{} // 获取信号量

		go func(start, end int) {
			defer func() {
				<-sem // 释放信号量
				wg.Done()
			}()

			if end > batchSize {
				end = batchSize
			}

			for b := start; b < end; b++ {
				startIdx := b * featureSize
				endIdx := startIdx + featureSize

				// 计算当前样本的sumSq
				sumSq := float32(0.0)
				for i := startIdx; i < endIdx; i++ {
					val := inputTensor.Data[i]
					sumSq += val * val
				}

				// 计算RMS
				meanSq := sumSq / float32(featureSize)
				rms := math.Sqrt(meanSq + r.eps)
				invRms := 1.0 / rms

				// 处理特征维度
				for i := startIdx; i < endIdx; i++ {
					featureIdx := i % featureSize
					output.Data[i] = inputTensor.Data[i] * invRms * r.Weights.Data[featureIdx]
				}
			}
		}(chunkStart, chunkStart+chunkSize)
	}

	wg.Wait()
	close(sem)

	return output
}

// Backward performs the RMSNorm backward pass.
func (r *RMSNorm) Backward(gradOutput *tensor.Tensor, learningRate float32) *tensor.Tensor {
	if r.inputCache == nil {
		panic("Forward pass must be called before backward pass")
	}
	inputTensor := r.inputCache

	// Initialize gradients
	gradInputShape := inputTensor.Shape
	gradInputData := make([]float32, len(inputTensor.Data))
	gradInput := tensor.NewTensor(gradInputData, gradInputShape)

	gradWeightsData := make([]float32, len(r.Weights.Data))

	// Calculate batch and feature sizes
	batchSize := product(inputTensor.Shape[:len(inputTensor.Shape)-1])
	featureSize := r.Weights.Shape[0]

	for b := 0; b < batchSize; b++ {
		start := b * featureSize
		end := start + featureSize

		// Recompute mean square and RMS
		sumSq := float32(0.0)
		for i := start; i < end; i++ {
			sumSq += inputTensor.Data[i] * inputTensor.Data[i]
		}
		meanSq := sumSq / float32(featureSize)
		rms := math.Sqrt(meanSq + r.eps)
		invRms := 1.0 / rms

		// Compute gradients
		for i := start; i < end; i++ {
			featureIdx := i % featureSize
			x := inputTensor.Data[i]

			// Gradient of output w.r.t. input
			dxHat := gradOutput.Data[i] * r.Weights.Data[featureIdx]
			gradInput.Data[i] = dxHat*invRms - (x*sumSq)/(float32(featureSize)*rms*rms*rms)

			// Gradient of output w.r.t. weights
			gradWeightsData[featureIdx] += gradOutput.Data[i] * (x * invRms)
		}
	}

	// Update weights
	for i := range r.Weights.Data {
		r.Weights.Data[i] -= learningRate * gradWeightsData[i] / float32(batchSize)
	}

	return gradInput
}

// ZeroGrad resets the gradients of weights to zero.
func (r *RMSNorm) ZeroGrad() {
	// No explicit gradient storage in this implementation
}

// Parameters returns a slice containing the weights tensor of the layer.
func (r *RMSNorm) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{r.Weights}
}

// SetWeights sets the weights of the layer.
func (r *RMSNorm) SetWeights(data []float32) {
	r.Weights = tensor.NewTensor(data, r.Weights.Shape)
}

// SetWeightsAndShape sets the weights and shape of the layer.
func (r *RMSNorm) SetWeightsAndShape(data []float32, shape []int) {
	r.Weights = tensor.NewTensor(data, shape)
}

func (r *RMSNorm) SetBias(data [][]float32) {
}

func (r *RMSNorm) SetBiasAndShape(data []float32, shape []int) {
}
