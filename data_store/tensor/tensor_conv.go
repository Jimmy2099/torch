package tensor

import (
	"errors"
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
)

func (t *Tensor) Dimensions() int {
	return len(t.shape)
}

func (t *Tensor) DimSize(dim int) int {
	if dim < 0 || dim >= len(t.shape) {
		panic("invalid dimension")
	}
	return t.shape[dim]
}

func (t *Tensor) Clone() *Tensor {
	newData := make([]float32, len(t.Data))
	copy(newData, t.Data)
	newShape := make([]int, len(t.shape))
	copy(newShape, t.shape)
	return &Tensor{
		Data:  newData,
		shape: newShape,
	}
}

func (t *Tensor) Multiply(other *Tensor) *Tensor {
	if !sameShape(t.shape, other.shape) {
		panic("Tensors must have the same shape for element-wise multiplication")
	}

	resultData := make([]float32, len(t.Data))
	for i := 0; i < len(t.Data); i++ {
		resultData[i] = t.Data[i] * other.Data[i]
	}

	return &Tensor{
		Data:  resultData,
		shape: t.shape,
	}
}

func (t *Tensor) Transpose() *Tensor {
	if len(t.shape) != 2 {
		panic("Transpose only works for 2D tensors")
	}

	rows := t.shape[0]
	cols := t.shape[1]

	newData := make([]float32, len(t.Data))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			newData[j*rows+i] = t.Data[i*cols+j]
		}
	}

	newShape := []int{cols, rows}
	//t.Data=newData
	//t.shape=newShape
	//return t
	return &Tensor{
		Data:  newData,
		shape: newShape,
	}
}

func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor{Data: %v, shape: %v}", t.Data, t.shape)
}

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

func (t *Tensor) GetSample(batchIdx int) *Tensor {
	if len(t.shape) != 4 {
		panic("GetSample only works for 4D tensors")
	}
	if batchIdx < 0 || batchIdx >= t.shape[0] {
		panic("invalid batch index")
	}

	channels, height, width := t.shape[1], t.shape[2], t.shape[3]
	sampleSize := channels * height * width
	data := make([]float32, sampleSize)

	copy(data, t.Data[batchIdx*sampleSize:(batchIdx+1)*sampleSize])

	return &Tensor{
		Data:  data,
		shape: []int{channels, height, width},
	}
}

func StackTensors(tensors []*Tensor, dim int) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, errors.New("empty tensor list")
	}

	baseShape := tensors[0].shape
	for _, t := range tensors {
		if !sameShape(t.shape, baseShape) {
			return nil, errors.New("all tensors must have the same shape")
		}
	}

	newShape := make([]int, len(baseShape)+1)
	copy(newShape, baseShape[:dim])
	newShape[dim] = len(tensors)
	copy(newShape[dim+1:], baseShape[dim:])

	elementSize := len(tensors[0].Data)
	newData := make([]float32, len(tensors)*elementSize)
	for i, t := range tensors {
		copy(newData[i*elementSize:(i+1)*elementSize], t.Data)
	}

	return &Tensor{
		Data:  newData,
		shape: newShape,
	}, nil
}

func (t *Tensor) im2col_get_pixel(row, col, channel, pad int) float32 {
	height, width := t.shape[1], t.shape[2]

	row -= pad
	col -= pad

	if row < 0 || col < 0 || row >= height || col >= width {
		return 0
	}

	return t.Data[col+width*(row+height*channel)]
}

func (t *Tensor) Pad2D(padH, padW int) *Tensor {
	if padH == 0 && padW == 0 {
		return t.Clone()
	}

	var batchSize, channels, height, width int
	var is4D bool

	switch len(t.shape) {
	case 3:
		channels, height, width = t.shape[0], t.shape[1], t.shape[2]
		is4D = false
	case 4:
		batchSize, channels, height, width = t.shape[0], t.shape[1], t.shape[2], t.shape[3]
		is4D = true
	default:
		panic("Pad2D only works for 3D or 4D tensors")
	}

	newHeight := height + 2*padH
	newWidth := width + 2*padW

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

	for b := 0; is4D && b < batchSize || !is4D && b < 1; b++ {
		for c := 0; c < channels; c++ {
			for i := 0; i < height; i++ {
				for j := 0; j < width; j++ {
					srcIdx := b*channels*height*width + c*height*width + i*width + j
					if !is4D {
						srcIdx = c*height*width + i*width + j
					}

					targetIdx := b*channels*newHeight*newWidth + c*newHeight*newWidth + (i+padH)*newWidth + (j + padW)
					if !is4D {
						targetIdx = c*newHeight*newWidth + (i+padH)*newWidth + (j + padW)
					}

					padded.Data[targetIdx] = t.Data[srcIdx]
				}
			}
		}
	}

	return padded
}

func (t *Tensor) Repeat(dim int, repeats int) *Tensor {
	if len(t.shape) != 2 && len(t.shape) != 4 {
		panic("Repeat currently only supports 2D or 4D tensors")
	}

	if len(t.shape) == 2 {
		rows, cols := t.shape[0], t.shape[1]
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
		batch, channels, height, width := t.shape[0], t.shape[1], t.shape[2], t.shape[3]
		var newData []float32
		var newShape []int

		switch dim {
		case 0:
			newData = make([]float32, batch*repeats*channels*height*width)
			newShape = []int{batch * repeats, channels, height, width}
			for r := 0; r < repeats; r++ {
				copy(newData[r*batch*channels*height*width:(r+1)*batch*channels*height*width],
					t.Data)
			}
		case 1:
			newData = make([]float32, batch*channels*repeats*height*width)
			newShape = []int{batch, channels * repeats, height, width}
			for b := 0; b < batch; b++ {
				for r := 0; r < repeats; r++ {
					copy(newData[b*channels*repeats*height*width+r*channels*height*width:b*channels*repeats*height*width+(r+1)*channels*height*width],
						t.Data[b*channels*height*width:(b+1)*channels*height*width])
				}
			}
		case 2, 3:
			panic("Repeating along spatial dimensions is not yet implemented")
		default:
			panic("Invalid dimension for 4D tensor")
		}
		return NewTensor(newData, newShape)
	}
}

func (t *Tensor) col2im(kernelSize, stride, pad, inHeight, inWidth int) *Tensor {
	if len(t.shape) != 2 {
		panic("input tensor must be 2D for col2im operation")
	}

	origHeight := inHeight + 2*pad
	origWidth := inWidth + 2*pad
	output := NewTensor(make([]float32, origHeight*origWidth), []int{origHeight, origWidth})

	for i := 0; i < t.shape[1]; i++ {
		h := (i / origWidth) * stride
		w := (i % origWidth) * stride
		patchData := t.GetCol(i)

		for dh := 0; dh < kernelSize; dh++ {
			for dw := 0; dw < kernelSize; dw++ {
				index := dh*kernelSize + dw
				if index >= len(patchData.Data) {
					continue
				}
				output.Data[h+dh+(w+dw)*origHeight] += patchData.Data[index]
			}
		}
	}

	cropped := output.Crop(pad)
	return cropped
}

func (t *Tensor) Pad(padding int) *Tensor {

	rows, cols := t.shape[0], t.shape[1]
	newRows := rows + 2*padding
	newCols := cols + 2*padding

	paddedData := make([]float32, newRows*newCols)
	padded := NewTensor(paddedData, []int{newRows, newCols})

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			padded.Data[(i+padding)*newCols+(j+padding)] = t.Data[i*cols+j]
		}
	}

	return padded
}

func (t *Tensor) Crop(padding int) *Tensor {
	if padding == 0 {
		return t
	}
	rows, cols := t.shape[0], t.shape[1]
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

func (t *Tensor) FlattenByDim(startDim, endDim int) *Tensor {
	if startDim < 0 || startDim >= len(t.shape) {
		panic("Invalid startDim")
	}
	if endDim < -1 || endDim >= len(t.shape) {
		panic("Invalid endDim")
	}

	if endDim == -1 {
		endDim = len(t.shape) - 1
	}

	rows := 1
	cols := 1

	for i := startDim; i <= endDim; i++ {
		rows *= t.shape[i]
	}

	for i := endDim + 1; i < len(t.shape); i++ {
		cols *= t.shape[i]
	}

	t.Reshape([]int{rows, cols})
	return t
}

func (t *Tensor) GetCols(start, end int) *Tensor {
	if len(t.shape) != 2 {
		panic("GetCols only works for 2D tensors")
	}
	if start < 0 || end > t.shape[1] || start >= end {
		panic("Invalid column range")
	}

	rows := t.shape[0]
	newCols := end - start
	resultData := make([]float32, rows*newCols)

	for i := 0; i < rows; i++ {
		for j := start; j < end; j++ {
			resultData[i*newCols+(j-start)] = t.Data[i*t.shape[1]+j]
		}
	}

	return NewTensor(resultData, []int{rows, newCols})
}

func (t *Tensor) SetCol(colIdx int, data *Tensor) {
	if len(t.shape) != 2 {
		panic("SetCol only works for 2D tensors")
	}
	if data.shape[0] != t.shape[0] || data.shape[1] != 1 {
		panic("Invalid column data dimensions")
	}

	for i := 0; i < t.shape[0]; i++ {
		t.Data[i*t.shape[1]+colIdx] = data.Data[i]
	}
}

func (t *Tensor) GetCol(colIdx int) *Tensor {
	if len(t.shape) != 2 {
		panic("GetCol only works for 2D tensors")
	}
	if colIdx < 0 || colIdx >= t.shape[1] {
		panic("Invalid column index")
	}

	rows := t.shape[0]
	resultData := make([]float32, rows)

	for i := 0; i < rows; i++ {
		resultData[i] = t.Data[i*t.shape[1]+colIdx]
	}
	return NewTensor(resultData, []int{rows})

}

func (t *Tensor) SumByDim(dim int) *Tensor {

	if len(t.shape) != 2 {
		panic("SumByDim works for 2D tensors")
	}

	if dim == 0 {
		resultData := make([]float32, t.shape[1])

		for j := 0; j < t.shape[1]; j++ {
			var sum float32
			for i := 0; i < t.shape[0]; i++ {
				sum += t.Data[i*t.shape[1]+j]
			}
			resultData[j] = sum
		}

		return NewTensor(resultData, []int{t.shape[1]})
	} else if dim == 1 {
		resultData := make([]float32, t.shape[0])

		for i := 0; i < t.shape[0]; i++ {
			var sum float32
			for j := 0; j < t.shape[1]; j++ {
				sum += t.Data[i*t.shape[1]+j]
			}
			resultData[i] = sum
		}

		return NewTensor(resultData, []int{t.shape[0]})

	}
	panic("Invalid dimension for sum")
}

func (t *Tensor) Expand(targetShape []int) *Tensor {
	if len(t.shape) != len(targetShape) {
		panic("expand dimensions must match")
	}

	for i := range t.shape {
		if t.shape[i] != 1 && t.shape[i] != targetShape[i] {
			panic(fmt.Sprintf("cannot expand dimension %d from %d to %d",
				i, t.shape[i], targetShape[i]))
		}
	}

	totalElements := 1
	for _, size := range targetShape {
		totalElements *= size
	}

	newData := make([]float32, totalElements)

	origStrides := make([]int, len(t.shape))
	targetStrides := make([]int, len(targetShape))
	origStride := 1
	targetStride := 1
	for i := len(t.shape) - 1; i >= 0; i-- {
		origStrides[i] = origStride
		targetStrides[i] = targetStride
		origStride *= t.shape[i]
		targetStride *= targetShape[i]
	}

	for i := 0; i < totalElements; i++ {
		origIndex := 0
		remaining := i
		for dim := len(targetShape) - 1; dim >= 0; dim-- {
			size := targetShape[dim]
			pos := remaining % size
			if t.shape[dim] != 1 {
				origIndex += pos * origStrides[dim]
			}
			remaining /= size
		}
		newData[i] = t.Data[origIndex]
	}

	return &Tensor{
		Data:  newData,
		shape: targetShape,
	}
}

func (t *Tensor) GetRow(row int) *Tensor {
	if len(t.shape) != 2 {
		panic("GetRow requires 2D tensor")
	}
	if row < 0 || row >= t.shape[0] {
		panic("row index out of range")
	}

	data := make([]float32, t.shape[1])
	copy(data, t.Data[row*t.shape[1]:(row+1)*t.shape[1]])

	return NewTensor(data, []int{1, t.shape[1]})
}

func (t *Tensor) Sigmoid() *Tensor {
	data := make([]float32, len(t.Data))
	for i, val := range t.Data {
		data[i] = 1.0 / (1.0 + math.Exp(-val))
	}
	return &Tensor{Data: data, shape: t.shape}
}

// Exp TODO TestCaseCheck
func (t *Tensor) Exp() *Tensor {
	newData := make([]float32, len(t.Data))

	for i, val := range t.Data {
		newData[i] = math.Exp(val)
	}

	newShape := make([]int, len(t.shape))
	copy(newShape, t.shape)

	return &Tensor{
		Data:  newData,
		shape: newShape,
	}
}

// Log TODO TestCaseCheck
func (t *Tensor) Log() *Tensor {
	if len(t.Data) == 0 {
		panic("tensor is empty")
	}

	logData := make([]float32, len(t.Data))

	for i, v := range t.Data {
		if v <= 0 {
			panic("tensor contains non-positive value(s)")
		}
		logData[i] = math.Log(v)
	}

	return &Tensor{
		Data:  logData,
		shape: t.shape,
	}
}

func (t *Tensor) Conv2D(weights *Tensor, kernelSize, stride, padH, padW int) *Tensor {
	var input *Tensor
	if len(t.shape) == 3 {
		input = t.Reshape([]int{1, t.shape[0], t.shape[1], t.shape[2]})
	} else if len(t.shape) == 4 {
		input = t.Clone()
	} else {
		panic("input tensor must be 3D or 4D")
	}

	batchSize := input.shape[0]
	inChannels := input.shape[1]
	height := input.shape[2]
	width := input.shape[3]

	outChannels := weights.shape[0]
	if weights.shape[1] != inChannels || weights.shape[2] != kernelSize || weights.shape[3] != kernelSize {
		panic("weights shape mismatch")
	}

	outHeight := (height-kernelSize+2*padH)/stride + 1
	outWidth := (width-kernelSize+2*padW)/stride + 1

	var padded *Tensor
	if padH > 0 || padW > 0 {
		padded = t.Pad2D(padH, padW)
	} else {
		padded = t.Clone()
	}

	output := NewTensor(make([]float32, batchSize*outChannels*outHeight*outWidth),
		[]int{batchSize, outChannels, outHeight, outWidth})

	for b := 0; b < batchSize; b++ {
		sample := padded.GetSample(b)
		unfolded, err := sample.im2col(kernelSize, kernelSize, stride, stride)
		if err != nil {
			panic(err)
		}

		reshapedWeights := weights.Reshape([]int{outChannels, kernelSize * kernelSize * inChannels})

		result := reshapedWeights.MatMul(unfolded)

		reshapedResult := result.Reshape([]int{outChannels, outHeight, outWidth})

		copy(output.Data[b*outChannels*outHeight*outWidth:], reshapedResult.Data)
	}

	return output
}

func (t *Tensor) Conv2DTranspose(weight *Tensor, kernelH, kernelW, strideH, strideW, padH, padW int) *Tensor {

	if len(t.GetShape()) != 4 || len(weight.GetShape()) != 4 {
		weight = weight.Reshape([]int{1, 1, kernelH, kernelW})
		//panic("both tensors must be 4D (NCHW format)")
	}

	transposedWeight := weight.TransposeByDim(0, 1)

	adjPadH := kernelH - padH - 1
	adjPadW := kernelW - padW - 1

	return t.Conv2D(transposedWeight, kernelH, strideH, adjPadH, adjPadW)
}

func (t *Tensor) Conv2DGradientWeight(input *Tensor, kernelH, kernelW, strideH, strideW, padH, padW int) *Tensor {
	fmt.Println()
	strideH = 2
	strideW = 2
	if len(t.GetShape()) != 4 || len(input.GetShape()) != 4 {
		panic("both tensors must be 4D (NCHW format)")
	}

	reshapedInput := input.TransposeByDim(0, 1)
	reshapedGrad := t.TransposeByDim(0, 1)

	kernelH = reshapedGrad.GetShape()[2]
	kernelW = reshapedGrad.GetShape()[3]

	fmt.Println("kernelH,kernelW:", kernelH, kernelW)

	return reshapedInput.Conv2D(reshapedGrad, kernelH, strideH, padH, padW)
}

//func (t *Tensor) Conv2DGradientWeight(input *Tensor, strideH, strideW, padH, padW int) *Tensor {
//	if len(t.GetShape()) != 4 || len(input.GetShape()) != 4 {
//		panic("both tensors must be 4D (NCHW format)")
//	}
//
//	reshapedInput := input.TransposeByDim(0, 1)
//	reshapedGrad := t.TransposeByDim(0, 1)
//
//	kernelH := reshapedGrad.GetShape()[2]
//	kernelW := reshapedGrad.GetShape()[3]
//
//	fmt.Println("kernelH,kernelW:", kernelH, kernelW)
//
//	return reshapedInput.Conv2D(reshapedGrad, kernelH, strideH, padH, padW)
//}

func (t *Tensor) im2col(kernelH, kernelW, strideH, strideW int) (*Tensor, error) {
	if len(t.GetShape()) != 3 {
		return nil, errors.New("im2col requires 3D tensor [C, H, W]")
	}

	channels := t.GetShape()[0]
	height := t.GetShape()[1]
	width := t.GetShape()[2]

	outHeight := (height-kernelH)/strideH + 1
	outWidth := (width-kernelW)/strideW + 1

	unfoldedSize := channels * kernelH * kernelW
	blocks := outHeight * outWidth

	data := make([]float32, unfoldedSize*blocks)
	idx := 0

	for c := 0; c < channels; c++ {
		for kh := 0; kh < kernelH; kh++ {
			for kw := 0; kw < kernelW; kw++ {
				for h := 0; h < outHeight; h++ {
					for w := 0; w < outWidth; w++ {
						srcH := h*strideH + kh
						srcW := w*strideW + kw
						if srcH >= height || srcW >= width {
							data[idx] = 0
						} else {
							srcIdx := c*height*width + srcH*width + srcW
							data[idx] = t.Data[srcIdx]
						}
						idx++
					}
				}
			}
		}
	}

	return &Tensor{
		Data:  data,
		shape: []int{unfoldedSize, blocks},
	}, nil
}

//func (t *Tensor) Conv2DGradientWeight(input *Tensor, kernelH, kernelW, strideH, strideW, padH, padW int) *Tensor {
//	reshapedInput := input.TransposeByDim(0, 1)
//	reshapedGrad := t.TransposeByDim(0, 1)
//	return reshapedInput.Conv2D(reshapedGrad, kernelH, strideH, padH, padW)
//}
