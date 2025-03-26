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
	data := make([]float64, sampleSize)

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
	newData := make([]float64, len(tensors)*elementSize)
	for i, t := range tensors {
		copy(newData[i*elementSize:(i+1)*elementSize], t.Data)
	}

	return &Tensor{
		Data:  newData,
		Shape: newShape,
	}, nil
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
			make([]float64, batchSize*channels*newHeight*newWidth),
			[]int{batchSize, channels, newHeight, newWidth},
		)
	} else {
		padded = NewTensor(
			make([]float64, channels*newHeight*newWidth),
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
		var newData []float64
		var newShape []int

		if dim == 0 {
			newData = make([]float64, rows*repeats*cols)
			newShape = []int{rows * repeats, cols}
			for r := 0; r < repeats; r++ {
				copy(newData[r*rows*cols:(r+1)*rows*cols], t.Data)
			}
		} else if dim == 1 {
			newData = make([]float64, rows*cols*repeats)
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
		var newData []float64
		var newShape []int

		switch dim {
		case 0: // Repeat along batch dimension
			newData = make([]float64, batch*repeats*channels*height*width)
			newShape = []int{batch * repeats, channels, height, width}
			for r := 0; r < repeats; r++ {
				copy(newData[r*batch*channels*height*width:(r+1)*batch*channels*height*width],
					t.Data)
			}
		case 1: // Repeat along channel dimension
			newData = make([]float64, batch*channels*repeats*height*width)
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
