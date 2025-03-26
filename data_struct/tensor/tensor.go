package tensor

import (
	"errors"
	"fmt"
)

type Tensor struct {
	Data  []float64
	Shape []int // e.g., [batch_size, channels, height, width]
}

func NewTensor(data []float64, shape []int) *Tensor {
	return &Tensor{Data: data, Shape: shape}
}

func (t *Tensor) Conv2D(weights *Tensor, kernelSize, stride, pad int) (*Tensor, error) {
	// Ensure input is 4D (add batch dimension if needed)
	var input *Tensor
	if len(t.Shape) == 3 {
		// Add batch dimension (1) to make it 4D
		input = t.Reshape([]int{1, t.Shape[0], t.Shape[1], t.Shape[2]})
	} else if len(t.Shape) == 4 {
		input = t.Clone()
	} else {
		return nil, errors.New("input tensor must be 3D or 4D")
	}

	batchSize := input.Shape[0]
	inChannels := input.Shape[1]
	height := input.Shape[2]
	width := input.Shape[3]

	outChannels := weights.Shape[0]
	if weights.Shape[1] != inChannels || weights.Shape[2] != kernelSize || weights.Shape[3] != kernelSize {
		return nil, errors.New("weights shape mismatch")
	}

	// Calculate output dimensions
	outHeight := (height-kernelSize+2*pad)/stride + 1
	outWidth := (width-kernelSize+2*pad)/stride + 1

	// Pad input if needed
	var padded *Tensor
	if pad > 0 {
		padded = t.Pad2D(pad)
	} else {
		padded = t.Clone()
	}

	// Initialize output tensor
	output := NewTensor(make([]float64, batchSize*outChannels*outHeight*outWidth),
		[]int{batchSize, outChannels, outHeight, outWidth})

	// Process each sample in batch
	for b := 0; b < batchSize; b++ {
		sample := padded.GetSample(b)
		unfolded, err := sample.im2col(kernelSize, stride)
		if err != nil {
			return nil, err
		}

		// Reshape weights
		reshapedWeights := weights.Reshape([]int{outChannels, kernelSize * kernelSize * inChannels})

		// Matrix multiplication
		result, err := reshapedWeights.MatMul(unfolded)
		if err != nil {
			return nil, err
		}

		// Reshape to output dimensions
		reshapedResult := result.Reshape([]int{outChannels, outHeight, outWidth})

		// Copy to output
		copy(output.Data[b*outChannels*outHeight*outWidth:], reshapedResult.Data)
	}

	return output, nil
}

// MatMul performs matrix multiplication between two 2D tensors
func (t *Tensor) MatMul(other *Tensor) (*Tensor, error) {
	if len(t.Shape) != 2 || len(other.Shape) != 2 {
		return nil, errors.New("both tensors must be 2D for matrix multiplication")
	}
	if t.Shape[1] != other.Shape[0] {
		return nil, fmt.Errorf("shape mismatch: %v and %v", t.Shape, other.Shape)
	}

	m, n := t.Shape[0], other.Shape[1]
	k := t.Shape[1]
	result := make([]float64, m*n)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for l := 0; l < k; l++ {
				sum += t.Data[i*k+l] * other.Data[l*n+j]
			}
			result[i*n+j] = sum
		}
	}

	return NewTensor(result, []int{m, n}), nil
}

// Expand expands the tensor to the target shape using broadcasting rules
func (t *Tensor) Expand(targetShape []int) *Tensor {
	if len(t.Shape) != len(targetShape) {
		panic("expand dimensions must match")
	}

	// Check if expansion is valid
	for i := range t.Shape {
		if t.Shape[i] != 1 && t.Shape[i] != targetShape[i] {
			panic(fmt.Sprintf("cannot expand dimension %d from %d to %d",
				i, t.Shape[i], targetShape[i]))
		}
	}

	// Calculate total elements in target shape
	totalElements := 1
	for _, size := range targetShape {
		totalElements *= size
	}

	// Create new data slice
	newData := make([]float64, totalElements)

	// Calculate strides for original and target shapes
	origStrides := make([]int, len(t.Shape))
	targetStrides := make([]int, len(targetShape))
	origStride := 1
	targetStride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		origStrides[i] = origStride
		targetStrides[i] = targetStride
		origStride *= t.Shape[i]
		targetStride *= targetShape[i]
	}

	// Fill new data using broadcasting
	for i := 0; i < totalElements; i++ {
		// Calculate original index
		origIndex := 0
		remaining := i
		for dim := len(targetShape) - 1; dim >= 0; dim-- {
			size := targetShape[dim]
			pos := remaining % size
			if t.Shape[dim] != 1 {
				origIndex += pos * origStrides[dim]
			}
			remaining /= size
		}
		newData[i] = t.Data[origIndex]
	}

	return &Tensor{
		Data:  newData,
		Shape: targetShape,
	}
}

// Conv2D implements a 2D convolution operation.
// It now supports input Tensor with shape [batch, channels, height, width] or [channels, height, width]
// and weights Tensor with shape [out_channels, in_channels, kernel_height, kernel_width].
func (t *Tensor) Conv2D1(weights *Tensor, kernelSize, stride, pad int) (*Tensor, error) {
	// Input dimension: (batch, channels, height, width) or (channels, height, width)
	// Weights dimension: (out_channels, in_channels, kernel_height, kernel_width)

	// Validate input shapes
	if len(t.Shape) != 3 && len(t.Shape) != 4 {
		return nil, errors.New("input tensor must have shape [batch, channels, height, width] or [channels, height, width]")
	}
	if len(weights.Shape) != 4 {
		return nil, errors.New("weights tensor must have shape [out_channels, in_channels, kernel_height, kernel_width]")
	}

	var batchSize, channels, height, width int
	if len(t.Shape) == 4 {
		// With batch dimension
		batchSize, channels, height, width = t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]
	} else {
		// Without batch dimension (treat as batch size 1)
		batchSize = 1
		channels, height, width = t.Shape[0], t.Shape[1], t.Shape[2]
	}

	outChannels, inChannels, kernelHeight, kernelWidth := weights.Shape[0], weights.Shape[1], weights.Shape[2], weights.Shape[3]

	if inChannels != channels {
		return nil, errors.New("input channels must match weight in_channels")
	}
	if kernelSize != kernelHeight || kernelSize != kernelWidth {
		return nil, errors.New("kernel size must be the same for both height and width in weights")
	}

	// Calculate output spatial dimensions
	outHeight := (height+2*pad-kernelSize)/stride + 1
	outWidth := (width+2*pad-kernelSize)/stride + 1

	// Process each sample in the batch
	var results []*Tensor
	for b := 0; b < batchSize; b++ {
		// Get current sample (with or without batch dimension)
		var sample *Tensor
		if len(t.Shape) == 4 {
			sample = t.GetSample(b) // Need to implement GetSample method
		} else {
			sample = t
		}

		// Pad the input
		paddedInput := sample.Pad2D(pad)

		// Perform im2col unfolding
		unfolded, err := paddedInput.im2col(kernelSize, stride)
		if err != nil {
			return nil, err
		}

		// Reshape weights
		reshapedWeights := weights.Reshape([]int{outChannels, kernelSize * kernelSize * inChannels})

		// Matrix multiplication
		result := reshapedWeights.Multiply(unfolded)

		// Reshape to output dimensions
		reshapedResult := result.Reshape([]int{outChannels, outHeight * outWidth})
		results = append(results, reshapedResult)
	}

	// Combine batch results
	if batchSize == 1 {
		return results[0], nil
	}

	// For batch size > 1, stack results along batch dimension
	return StackTensors(results, 0) // Need to implement StackTensors function
}

// MatMul 矩阵乘法
func (t *Tensor) MatMul1(other *Tensor) *Tensor {
	// 简单实现，假设是2D矩阵
	if len(t.Shape) != 2 || len(other.Shape) != 2 {
		panic("MatMul requires 2D tensors")
	}
	if t.Shape[1] != other.Shape[0] {
		panic("matrix dimensions mismatch")
	}

	m, n := t.Shape[0], other.Shape[1]
	k := t.Shape[1]
	result := make([]float64, m*n)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for l := 0; l < k; l++ {
				sum += t.Data[i*k+l] * other.Data[l*n+j]
			}
			result[i*n+j] = sum
		}
	}

	return &Tensor{Data: result, Shape: []int{m, n}}
}
