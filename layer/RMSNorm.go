package layer

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
	"runtime"
	"sync"
)

type RMSNorm struct {
	Weights    *tensor.Tensor
	eps        float32
	inputCache *tensor.Tensor
}

func (r *RMSNorm) GetWeights() *tensor.Tensor {
	return r.Weights
}

func (r *RMSNorm) GetBias() *tensor.Tensor {
	return nil
}

func NewRMSNorm(features int, eps float32) *RMSNorm {
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

func (r *RMSNorm) Backward(gradOutput *tensor.Tensor, learningRate float32) *tensor.Tensor {
	return nil

}

func (r *RMSNorm) ZeroGrad() {
}

func (r *RMSNorm) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{r.Weights}
}

func (r *RMSNorm) SetWeights(data []float32) {
	r.Weights = tensor.NewTensor(data, r.Weights.GetShape())
}

func (r *RMSNorm) SetWeightsAndShape(data []float32, shape []int) {
	r.Weights = tensor.NewTensor(data, shape)
}

func (r *RMSNorm) SetBias(data [][]float32) {
}

func (r *RMSNorm) SetBiasAndShape(data []float32, shape []int) {
}

var numCPU = runtime.NumCPU()

func (r *RMSNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	return r.ForwardSignalThread(x)
}

func (r *RMSNorm) ForwardSignalThread(inputTensor *tensor.Tensor) *tensor.Tensor {
	if len(inputTensor.GetShape()) == 1 {
		inputTensor = inputTensor.Reshape([]int{1, inputTensor.GetShape()[0]})
	}
	if len(inputTensor.GetShape()) < 2 {
		panic(fmt.Sprintf("Input must be at least 2D tensor, got shape: %v", inputTensor.GetShape()))
	}

	features := inputTensor.GetShape()[len(inputTensor.GetShape())-1]
	if features != r.Weights.GetShape()[0] {
		panic(fmt.Sprintf("Feature dimension mismatch: input has %d, weights have %d", features, r.Weights.GetShape()[0]))
	}

	r.inputCache = inputTensor

	outputShape := inputTensor.GetShape()
	outputData := make([]float32, len(inputTensor.Data))
	output := tensor.NewTensor(outputData, outputShape)

	batchSize := product(inputTensor.GetShape()[:len(inputTensor.GetShape())-1])
	featureSize := features

	for b := 0; b < batchSize; b++ {
		start := b * featureSize
		end := start + featureSize

		sumSq := float32(0.0)
		for i := start; i < end; i++ {
			sumSq += inputTensor.Data[i] * inputTensor.Data[i]
		}
		meanSq := sumSq / float32(featureSize)

		rms := math.Sqrt(meanSq + r.eps)

		for i := start; i < end; i++ {
			featureIdx := i % featureSize
			output.Data[i] = (inputTensor.Data[i] / rms) * r.Weights.Data[featureIdx]
		}
	}

	return output
}

func (r *RMSNorm) ForwardMultiThread(inputTensor *tensor.Tensor) *tensor.Tensor {
	if len(inputTensor.GetShape()) == 1 {
		inputTensor = inputTensor.Reshape([]int{1, inputTensor.GetShape()[0]})
	}

	if len(inputTensor.GetShape()) < 2 {
		panic(fmt.Sprintf("Input must be at least 2D tensor, got shape: %v", inputTensor.GetShape()))
	}

	features := inputTensor.GetShape()[len(inputTensor.GetShape())-1]
	if features != r.Weights.GetShape()[0] {
		panic(fmt.Sprintf("Feature dimension mismatch: input has %d, weights have %d", features, r.Weights.GetShape()[0]))
	}

	r.inputCache = inputTensor

	outputShape := inputTensor.GetShape()
	outputData := make([]float32, len(inputTensor.Data))
	output := tensor.NewTensor(outputData, outputShape)

	batchSize := product(inputTensor.GetShape()[:len(inputTensor.GetShape())-1])
	featureSize := features

	chunkSize := (batchSize + numCPU - 1) / numCPU
	var wg sync.WaitGroup

	sem := make(chan struct{}, numCPU*2)

	for chunkStart := 0; chunkStart < batchSize; chunkStart += chunkSize {
		wg.Add(1)
		sem <- struct{}{}

		go func(start, end int) {
			defer func() {
				<-sem
				wg.Done()
			}()

			if end > batchSize {
				end = batchSize
			}

			for b := start; b < end; b++ {
				startIdx := b * featureSize
				endIdx := startIdx + featureSize

				sumSq := float32(0.0)
				for i := startIdx; i < endIdx; i++ {
					val := inputTensor.Data[i]
					sumSq += val * val
				}

				meanSq := sumSq / float32(featureSize)
				rms := math.Sqrt(meanSq + r.eps)
				invRms := 1.0 / rms

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
