package tensor

import (
	"errors"
	"fmt"
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
	newData := make([]float64, len(t.Data))
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

	resultData := make([]float64, len(t.Data))
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

	newData := make([]float64, len(t.Data))
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

// Conv2D implements a 2D convolution operation.
// It assumes the input Tensor has shape [channels, height, width] and the weights
// Tensor has shape [out_channels, in_channels, kernel_height, kernel_width].
func (t *Tensor) Conv2D(weights *Tensor, kernelSize, stride, pad int) (*Tensor, error) {
	// Input dimension: (channels, height, width)
	// Weights dimension: (out_channels, in_channels, kernel_height, kernel_width)

	// Validate input shapes
	if len(t.Shape) != 3 {
		return nil, errors.New("input tensor must have shape [channels, height, width]")
	}
	if len(weights.Shape) != 4 {
		return nil, errors.New("weights tensor must have shape [out_channels, in_channels, kernel_height, kernel_width]")
	}

	channels, height, width := t.Shape[0], t.Shape[1], t.Shape[2]
	outChannels, inChannels, kernelHeight, kernelWidth := weights.Shape[0], weights.Shape[1], weights.Shape[2], weights.Shape[3]

	if inChannels != channels {
		return nil, errors.New("input channels must match weight in_channels")
	}
	if kernelSize != kernelHeight || kernelSize != kernelWidth {
		return nil, errors.New("Kernel size must be the same for both height and width in weights")
	}

	// Calculate output spatial dimensions
	outHeight := (height+2*pad-kernelSize)/stride + 1
	outWidth := (width+2*pad-kernelSize)/stride + 1

	// Pad the input
	paddedInput := t.Pad2D(pad)

	// Perform im2col unfolding
	unfolded, err := paddedInput.im2col(kernelSize, stride)
	if err != nil {
		return nil, err
	}

	// Reshape weights for matrix multiplication (out_channels, kernel_size * kernel_size * in_channels)
	reshapedWeights := weights.Reshape([]int{outChannels, kernelSize * kernelSize * inChannels})

	// Matrix multiplication: reshaped_weights * unfolded
	result := reshapedWeights.Multiply(unfolded)

	// Reshape to output dimensions
	return result.Reshape([]int{outChannels, outHeight * outWidth}), nil
}

// im2col_get_pixel implements boundary check for pixel access.
func (t *Tensor) im2col_get_pixel(row, col, channel, pad int) float64 {
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
	channels, height, width := t.Shape[0], t.Shape[1], t.Shape[2]

	heightCol := (height-kernelSize)/stride + 1
	widthCol := (width-kernelSize)/stride + 1
	channelsCol := channels * kernelSize * kernelSize

	cols := NewTensor(make([]float64, channelsCol*heightCol*widthCol), []int{channelsCol, heightCol * widthCol})
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

// Pad2D adds 2D padding to the tensor.
func (t *Tensor) Pad2D(pad int) *Tensor {
	if pad == 0 {
		return t.Clone()
	}

	channels, size, _ := t.Shape[0], t.Shape[1], t.Shape[2] // assuming a square tensor for simplicity
	newSize := size + 2*pad

	padded := NewTensor(make([]float64, channels*newSize*newSize), []int{channels, newSize, newSize})

	for c := 0; c < channels; c++ {
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				padded.Data[(i+pad)*newSize+(j+pad)+c*newSize*newSize] = t.Data[i*size+j+c*size*size]
			}
		}
	}

	return padded
}

// Repeat duplicates the tensor along rows and columns.
func (t *Tensor) Repeat(rowRepeat, colRepeat int) *Tensor {
	if len(t.Shape) != 2 {
		panic("Repeat only works for 2D tensors")
	}
	rows, cols := t.Shape[0], t.Shape[1]

	newRows := rows * rowRepeat
	newCols := cols * colRepeat
	result := NewTensor(make([]float64, newRows*newCols), []int{newRows, newCols})

	for i := 0; i < newRows; i++ {
		for j := 0; j < newCols; j++ {
			result.Data[i*newCols+j] = t.Data[(i%rows)*cols+(j%cols)]
		}
	}
	return result
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

	output := NewTensor(make([]float64, origHeight*origWidth), []int{origHeight, origWidth})

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

	paddedData := make([]float64, newRows*newCols)
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

	croppedData := make([]float64, newRows*newCols)
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
	resultData := make([]float64, rows*newCols)

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
	resultData := make([]float64, rows)

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
		resultData := make([]float64, t.Shape[1])

		for j := 0; j < t.Shape[1]; j++ {
			sum := 0.0
			for i := 0; i < t.Shape[0]; i++ {
				sum += t.Data[i*t.Shape[1]+j]
			}
			resultData[j] = sum
		}

		return NewTensor(resultData, []int{t.Shape[1]})
	} else if dim == 1 { // Sum along columns, returns a row vector
		resultData := make([]float64, t.Shape[0])

		for i := 0; i < t.Shape[0]; i++ {
			sum := 0.0
			for j := 0; j < t.Shape[1]; j++ {
				sum += t.Data[i*t.Shape[1]+j]
			}
			resultData[i] = sum
		}

		return NewTensor(resultData, []int{t.Shape[0]})

	}
	panic("Invalid dimension for sum")
}
