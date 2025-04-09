package tensor

import (
	"errors"
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
)

// Dimensions returns the number of dimensions of the tensor.
func (t *Tensor) Dimensions() int {
	return len(t.Shape)
}

// DimSize returns the size of a specific dimension.
func (t *Tensor) DimSize(dim int) int {
	if dim < 0 || dim >= len(t.Shape) {
		panic("invalid dimension")
	}
	return t.Shape[dim]
}

// Clone creates a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	newData := make([]float32, len(t.Data))
	copy(newData, t.Data)
	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	return &Tensor{
		Data:  newData,
		Shape: newShape,
	}
}

// Multiply performs element-wise multiplication with another tensor.
func (t *Tensor) Multiply(other *Tensor) *Tensor {
	if !sameShape(t.Shape, other.Shape) {
		panic("Tensors must have the same shape for element-wise multiplication")
	}

	resultData := make([]float32, len(t.Data))
	for i := 0; i < len(t.Data); i++ {
		resultData[i] = t.Data[i] * other.Data[i]
	}

	return &Tensor{
		Data:  resultData,
		Shape: t.Shape,
	}
}

// Transpose transposes the tensor (only works for 2D tensors).
func (t *Tensor) Transpose() *Tensor {
	if len(t.Shape) != 2 {
		panic("Transpose only works for 2D tensors")
	}

	rows := t.Shape[0]
	cols := t.Shape[1]

	newData := make([]float32, len(t.Data))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			newData[j*rows+i] = t.Data[i*cols+j]
		}
	}

	newShape := []int{cols, rows}

	return &Tensor{
		Data:  newData,
		Shape: newShape,
	}
}

func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor{Data: %v, Shape: %v}", t.Data, t.Shape)
}

// sameShape checks if two shapes are the same.
func sameShape(shape1, shape2 []int) bool {
	if len(shape1) != len(shape2) {
		return false
	}
	for i := 0; i < len(shape1); i++ {
		if shape1[i] != shape2[i] {
			return false
		}
	}
	return true
}

// GetSample extracts a single sample from a batched tensor
func (t *Tensor) GetSample(batchIdx int) *Tensor {
	if len(t.Shape) != 4 {
		panic("GetSample only works for 4D tensors")
	}
	if batchIdx < 0 || batchIdx >= t.Shape[0] {
		panic("invalid batch index")
	}

	channels, height, width := t.Shape[1], t.Shape[2], t.Shape[3]
	sampleSize := channels * height * width
	data := make([]float32, sampleSize)

	copy(data, t.Data[batchIdx*sampleSize:(batchIdx+1)*sampleSize])

	return &Tensor{
		Data:  data,
		Shape: []int{channels, height, width},
	}
}

// StackTensors combines tensors along a new dimension
func StackTensors(tensors []*Tensor, dim int) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, errors.New("empty tensor list")
	}

	// Validate all tensors have same shape
	baseShape := tensors[0].Shape
	for _, t := range tensors {
		if !sameShape(t.Shape, baseShape) {
			return nil, errors.New("all tensors must have the same shape")
		}
	}

	// Calculate new shape
	newShape := make([]int, len(baseShape)+1)
	copy(newShape, baseShape[:dim])
	newShape[dim] = len(tensors)
	copy(newShape[dim+1:], baseShape[dim:])

	// Combine data
	elementSize := len(tensors[0].Data)
	newData := make([]float32, len(tensors)*elementSize)
	for i, t := range tensors {
		copy(newData[i*elementSize:(i+1)*elementSize], t.Data)
	}

	return &Tensor{
		Data:  newData,
		Shape: newShape,
	}, nil
}

// im2col_get_pixel implements boundary check for pixel access.
func (t *Tensor) im2col_get_pixel(row, col, channel, pad int) float32 {
	height, width := t.Shape[1], t.Shape[2]

	row -= pad
	col -= pad

	if row < 0 || col < 0 || row >= height || col >= width {
		return 0
	}

	return t.Data[col+width*(row+height*channel)]
}

// im2col unfolds the input tensor into columns.
func (t *Tensor) im2col(kernelSize, stride int) (*Tensor, error) {
	// Handle both 3D and 4D tensors
	var channels, height, width int
	if len(t.Shape) == 3 {
		channels, height, width = t.Shape[0], t.Shape[1], t.Shape[2]
	} else if len(t.Shape) == 4 {
		channels, height, width = t.Shape[1], t.Shape[2], t.Shape[3]
	} else {
		return nil, errors.New("input tensor must be 3D or 4D")
	}

	heightCol := (height-kernelSize)/stride + 1
	widthCol := (width-kernelSize)/stride + 1
	channelsCol := channels * kernelSize * kernelSize

	cols := NewTensor(make([]float32, channelsCol*heightCol*widthCol), []int{channelsCol, heightCol * widthCol})
	dataCol := cols.Data

	for c := 0; c < channelsCol; c++ {
		wOffset := c % kernelSize
		hOffset := (c / kernelSize) % kernelSize
		cIm := c / kernelSize / kernelSize

		for h := 0; h < heightCol; h++ {
			for w := 0; w < widthCol; w++ {
				imRow := hOffset + h*stride
				imCol := wOffset + w*stride
				colIndex := (c*heightCol+h)*widthCol + w
				dataCol[colIndex] = t.im2col_get_pixel(imRow, imCol, cIm, 0)
			}
		}
	}

	return cols, nil
}

func (t *Tensor) Pad2D(pad int) *Tensor {
	if pad == 0 {
		return t.Clone()
	}

	// Determine tensor dimensions
	var batchSize, channels, height, width int
	var is4D bool

	switch len(t.Shape) {
	case 3:
		// 3D tensor: (channels, height, width)
		channels, height, width = t.Shape[0], t.Shape[1], t.Shape[2]
		is4D = false
	case 4:
		// 4D tensor: (batch, channels, height, width)
		batchSize, channels, height, width = t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]
		is4D = true
	default:
		panic("Pad2D only works for 3D or 4D tensors")
	}

	newHeight := height + 2*pad
	newWidth := width + 2*pad

	// Create padded tensor
	var padded *Tensor
	if is4D {
		padded = NewTensor(
			make([]float32, batchSize*channels*newHeight*newWidth),
			[]int{batchSize, channels, newHeight, newWidth},
		)
	} else {
		padded = NewTensor(
			make([]float32, channels*newHeight*newWidth),
			[]int{channels, newHeight, newWidth},
		)
	}

	// Apply padding
	for b := 0; is4D && b < batchSize || !is4D && b < 1; b++ {
		for c := 0; c < channels; c++ {
			for i := 0; i < height; i++ {
				for j := 0; j < width; j++ {
					// Calculate source and target indices
					srcIdx := b*channels*height*width + c*height*width + i*width + j
					if !is4D {
						srcIdx = c*height*width + i*width + j
					}

					targetIdx := b*channels*newHeight*newWidth + c*newHeight*newWidth + (i+pad)*newWidth + (j + pad)
					if !is4D {
						targetIdx = c*newHeight*newWidth + (i+pad)*newWidth + (j + pad)
					}

					padded.Data[targetIdx] = t.Data[srcIdx]
				}
			}
		}
	}

	return padded
}

// Repeat duplicates the tensor along rows and columns.
// Repeat repeats the tensor along specified dimensions (supports 2D and 4D tensors)
func (t *Tensor) Repeat(dim int, repeats int) *Tensor {
	if len(t.Shape) != 2 && len(t.Shape) != 4 {
		panic("Repeat currently only supports 2D or 4D tensors")
	}

	if len(t.Shape) == 2 {
		// Original 2D implementation
		rows, cols := t.Shape[0], t.Shape[1]
		var newData []float32
		var newShape []int

		if dim == 0 {
			newData = make([]float32, rows*repeats*cols)
			newShape = []int{rows * repeats, cols}
			for r := 0; r < repeats; r++ {
				copy(newData[r*rows*cols:(r+1)*rows*cols], t.Data)
			}
		} else if dim == 1 {
			newData = make([]float32, rows*cols*repeats)
			newShape = []int{rows, cols * repeats}
			for i := 0; i < rows; i++ {
				for r := 0; r < repeats; r++ {
					copy(newData[i*cols*repeats+r*cols:(i*cols*repeats)+(r+1)*cols],
						t.Data[i*cols:(i+1)*cols])
				}
			}
		} else {
			panic("Invalid dimension for 2D tensor")
		}
		return NewTensor(newData, newShape)
	} else {
		// 4D tensor implementation (batch, channels, height, width)
		batch, channels, height, width := t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]
		var newData []float32
		var newShape []int

		switch dim {
		case 0: // Repeat along batch dimension
			newData = make([]float32, batch*repeats*channels*height*width)
			newShape = []int{batch * repeats, channels, height, width}
			for r := 0; r < repeats; r++ {
				copy(newData[r*batch*channels*height*width:(r+1)*batch*channels*height*width],
					t.Data)
			}
		case 1: // Repeat along channel dimension
			newData = make([]float32, batch*channels*repeats*height*width)
			newShape = []int{batch, channels * repeats, height, width}
			for b := 0; b < batch; b++ {
				for r := 0; r < repeats; r++ {
					copy(newData[b*channels*repeats*height*width+r*channels*height*width:b*channels*repeats*height*width+(r+1)*channels*height*width],
						t.Data[b*channels*height*width:(b+1)*channels*height*width])
				}
			}
		case 2, 3: // Repeat along spatial dimensions (not commonly needed)
			panic("Repeating along spatial dimensions is not yet implemented")
		default:
			panic("Invalid dimension for 4D tensor")
		}
		return NewTensor(newData, newShape)
	}
}

// Conv2DGradWeights calculates the gradient of the weights in a convolution operation.
func (t *Tensor) Conv2DGradWeights(gradOutput *Tensor, kernelSize, stride, pad int) (*Tensor, error) {
	// Input gradient dimension: (out_channels, out_height*out_width)
	// Output gradient dimension: (out_channels, in_channels*kernelSize*kernelSize)

	//channels, height, width := t.Shape[0], t.Shape[1], t.Shape[2]

	// Unfold the input
	unfolded, err := t.im2col(kernelSize, stride)
	if err != nil {
		return nil, err
	}

	// Matrix multiplication: gradOutput * unfolded^T
	return gradOutput.Multiply(unfolded.Transpose()), nil
}

// Conv2DGradInput calculates the gradient of the input in a convolution operation.
func (t *Tensor) Conv2DGradInput(weights *Tensor, kernelSize, stride, pad int) (*Tensor, error) {
	// Input gradient dimension: (out_channels, out_height*out_width)
	// Output gradient dimension: (in_channels, in_height*in_width)

	// Transpose weights matrix
	wT := weights.Transpose()

	// Matrix multiplication: wT * gradOutput
	result := wT.Multiply(t)

	// Perform col2im operation
	return result.col2im(kernelSize, stride, pad, t.Shape[1], t.Shape[2])
}

// col2im rearranges the unfolded columns back into an image format.
func (t *Tensor) col2im(kernelSize, stride, pad, inHeight, inWidth int) (*Tensor, error) {
	// Input: (out_channels, out_height * out_width)
	// Output: (in_channels, in_height, in_width)

	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("input tensor must be 2D for col2im operation")
	}

	//outChannels := t.Shape[0]
	origHeight := inHeight + 2*pad
	origWidth := inWidth + 2*pad

	output := NewTensor(make([]float32, origHeight*origWidth), []int{origHeight, origWidth})

	for i := 0; i < t.Shape[1]; i++ {
		h := (i / origWidth) * stride
		w := (i % origWidth) * stride

		// Extract patch from unfolded tensor
		patchData := t.GetCol(i) // Assumes GetCol returns 1D tensor

		for dh := 0; dh < kernelSize; dh++ {
			for dw := 0; dw < kernelSize; dw++ {
				index := dh*kernelSize + dw
				if index < len(patchData.Data) {
					output.Data[h+dh+(w+dw)*origHeight] += patchData.Data[index]
				} else {
					fmt.Printf("index out of bounds %d \n", index) //debug statement - needs deletion
				}

			}
		}
	}

	cropped := output.Crop(pad)

	return cropped, nil
}

// Pad adds padding around the tensor.
func (t *Tensor) Pad(padding int) *Tensor {

	rows, cols := t.Shape[0], t.Shape[1]
	newRows := rows + 2*padding
	newCols := cols + 2*padding

	paddedData := make([]float32, newRows*newCols)
	padded := NewTensor(paddedData, []int{newRows, newCols})

	// Copy original data to the center of the padded tensor
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			padded.Data[(i+padding)*newCols+(j+padding)] = t.Data[i*cols+j]
		}
	}

	return padded
}

// Crop removes padding around the tensor.
func (t *Tensor) Crop(padding int) *Tensor {
	if padding == 0 {
		return t
	}
	rows, cols := t.Shape[0], t.Shape[1]
	newRows := rows - 2*padding
	newCols := cols - 2*padding

	croppedData := make([]float32, newRows*newCols)
	cropped := NewTensor(croppedData, []int{newRows, newCols})

	for i := 0; i < newRows; i++ {
		for j := 0; j < newCols; j++ {
			cropped.Data[i*newCols+j] = t.Data[(i+padding)*cols+(j+padding)]
		}
	}

	return cropped
}

// FlattenByDim flattens the tensor from startDim to endDim.
func (t *Tensor) FlattenByDim(startDim, endDim int) *Tensor {
	if startDim < 0 || startDim >= len(t.Shape) {
		panic("Invalid startDim")
	}
	if endDim < -1 || endDim >= len(t.Shape) {
		panic("Invalid endDim")
	}

	if endDim == -1 {
		endDim = len(t.Shape) - 1 // -1 means the last dimension
	}

	rows := 1
	cols := 1

	// Calculate the flattened dimensions from startDim to endDim
	for i := startDim; i <= endDim; i++ {
		rows *= t.Shape[i]
	}

	// Calculate the remaining dimensions
	for i := endDim + 1; i < len(t.Shape); i++ {
		cols *= t.Shape[i]
	}

	// Reshape
	t.Reshape([]int{rows, cols})
	return t
}

// GetCols returns a sub-tensor containing the specified column range.
func (t *Tensor) GetCols(start, end int) *Tensor {
	if len(t.Shape) != 2 {
		panic("GetCols only works for 2D tensors")
	}
	if start < 0 || end > t.Shape[1] || start >= end {
		panic("Invalid column range")
	}

	rows := t.Shape[0]
	newCols := end - start
	resultData := make([]float32, rows*newCols)

	for i := 0; i < rows; i++ {
		for j := start; j < end; j++ {
			resultData[i*newCols+(j-start)] = t.Data[i*t.Shape[1]+j]
		}
	}

	return NewTensor(resultData, []int{rows, newCols})
}

// SetCol sets the data of a specified column.
func (t *Tensor) SetCol(colIdx int, data *Tensor) {
	if len(t.Shape) != 2 {
		panic("SetCol only works for 2D tensors")
	}
	if data.Shape[0] != t.Shape[0] || data.Shape[1] != 1 {
		panic("Invalid column data dimensions")
	}

	for i := 0; i < t.Shape[0]; i++ {
		t.Data[i*t.Shape[1]+colIdx] = data.Data[i]
	}
}

// GetCol returns a column as a Tensor.
func (t *Tensor) GetCol(colIdx int) *Tensor {
	if len(t.Shape) != 2 {
		panic("GetCol only works for 2D tensors")
	}
	if colIdx < 0 || colIdx >= t.Shape[1] {
		panic("Invalid column index")
	}

	rows := t.Shape[0]
	resultData := make([]float32, rows)

	for i := 0; i < rows; i++ {
		resultData[i] = t.Data[i*t.Shape[1]+colIdx]
	}
	return NewTensor(resultData, []int{rows})

}

// SumByDim calculates the sum along a specified dimension.
func (t *Tensor) SumByDim(dim int) *Tensor {

	if len(t.Shape) != 2 {
		panic("SumByDim works for 2D tensors")
	}

	if dim == 0 { // Sum along rows, returns a column vector
		resultData := make([]float32, t.Shape[1])

		for j := 0; j < t.Shape[1]; j++ {
			var sum float32
			for i := 0; i < t.Shape[0]; i++ {
				sum += t.Data[i*t.Shape[1]+j]
			}
			resultData[j] = sum
		}

		return NewTensor(resultData, []int{t.Shape[1]})
	} else if dim == 1 { // Sum along columns, returns a row vector
		resultData := make([]float32, t.Shape[0])

		for i := 0; i < t.Shape[0]; i++ {
			var sum float32
			for j := 0; j < t.Shape[1]; j++ {
				sum += t.Data[i*t.Shape[1]+j]
			}
			resultData[i] = sum
		}

		return NewTensor(resultData, []int{t.Shape[0]})

	}
	panic("Invalid dimension for sum")
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
	output := NewTensor(make([]float32, batchSize*outChannels*outHeight*outWidth),
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
		result := reshapedWeights.MatMul(unfolded)

		// Reshape to output dimensions
		reshapedResult := result.Reshape([]int{outChannels, outHeight, outWidth})

		// Copy to output
		copy(output.Data[b*outChannels*outHeight*outWidth:], reshapedResult.Data)
	}

	return output, nil
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
	newData := make([]float32, totalElements)

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

func (t *Tensor) GetRow(row int) *Tensor {
	if len(t.Shape) != 2 {
		panic("GetRow requires 2D tensor")
	}
	if row < 0 || row >= t.Shape[0] {
		panic("row index out of range")
	}

	data := make([]float32, t.Shape[1])
	copy(data, t.Data[row*t.Shape[1]:(row+1)*t.Shape[1]])

	return NewTensor(data, []int{1, t.Shape[1]})
}

func (t *Tensor) Sigmoid() *Tensor {
	data := make([]float32, len(t.Data))
	for i, val := range t.Data {
		data[i] = 1.0 / (1.0 + math.Exp(-val))
	}
	return &Tensor{Data: data, Shape: t.Shape}
}
