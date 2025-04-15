package tensor

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
)

func (t *Tensor) TransposeByDim(dim1, dim2 int) *Tensor {
	if len(t.Shape) < 2 {
		panic("Tensor must have at least 2 dimensions to transpose")
	}
	if dim1 < 0 || dim1 >= len(t.Shape) || dim2 < 0 || dim2 >= len(t.Shape) {
		panic("Invalid transpose dimensions")
	}

	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[dim1], newShape[dim2] = newShape[dim2], newShape[dim1]

	oldStrides := computeStrides(t.Shape)
	newStrides := computeStrides(newShape)

	newData := make([]float32, len(t.Data))
	for i := range newData {
		indices := make([]int, len(t.Shape))
		remaining := i
		for d := 0; d < len(newShape); d++ {
			indices[d] = remaining / newStrides[d]
			remaining = remaining % newStrides[d]
		}

		indices[dim1], indices[dim2] = indices[dim2], indices[dim1]

		oldIndex := 0
		for d := 0; d < len(t.Shape); d++ {
			oldIndex += indices[d] * oldStrides[d]
		}

		newData[i] = t.Data[oldIndex]
	}

	return &Tensor{
		Data:  newData,
		Shape: newShape,
	}
}

func (t *Tensor) SplitLastDim(splitPoint, part int) *Tensor {
	if len(t.Shape) == 0 {
		panic("cannot split an empty dimension")
	}
	lastDim := t.Shape[len(t.Shape)-1]
	if splitPoint <= 0 || splitPoint >= lastDim {
		panic("invalid split point")
	}

	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[len(newShape)-1] = splitPoint

	start := part * splitPoint
	end := start + splitPoint
	if end > lastDim {
		end = lastDim
	}

	data := make([]float32, product(newShape))
	stride := product(t.Shape[len(t.Shape)-1:])
	_ = stride

	for i := 0; i < product(t.Shape[:len(t.Shape)-1]); i++ {
		srcStart := i*lastDim + start
		srcEnd := i*lastDim + end
		dstStart := i * splitPoint
		copy(data[dstStart:dstStart+splitPoint], t.Data[srcStart:srcEnd])
	}

	return &Tensor{
		Data:  data,
		Shape: newShape,
	}
}

func (t *Tensor) Slice(start, end, dim int) *Tensor {
	if t == nil {
		panic("attempting to slice an empty tensor")
	}
	ndim := len(t.Shape)
	if dim < 0 || dim >= ndim {
		panic(fmt.Sprintf("invalid slice dimension %d (total dimensions %d)", dim, ndim))
	}
	if start < 0 || end > t.Shape[dim] || start >= end {
		panic(fmt.Sprintf("invalid slice range [%d:%d] in dimension %d (length %d)",
			start, end, dim, t.Shape[dim]))
	}

	newShape := make([]int, ndim)
	copy(newShape, t.Shape)
	newShape[dim] = end - start

	strides := computeStrides(t.Shape)
	_ = strides

	newSize := product(newShape)
	newData := make([]float32, newSize)

	sliceDimSize := t.Shape[dim]
	outerDims := product(t.Shape[:dim])
	innerDims := product(t.Shape[dim+1:])

	for outer := 0; outer < outerDims; outer++ {
		outerOffset := outer * sliceDimSize * innerDims

		sliceStart := outerOffset + start*innerDims
		sliceEnd := outerOffset + end*innerDims

		copySize := (end - start) * innerDims
		copy(newData[outer*copySize:], t.Data[sliceStart:sliceEnd])
	}

	return &Tensor{
		Data:  newData,
		Shape: newShape,
	}
}

func (t *Tensor) Concat(other *Tensor, dim int) *Tensor {
	if t == nil || other == nil {
		panic("input tensor cannot be empty")
	}
	if len(t.Shape) != len(other.Shape) {
		panic(fmt.Sprintf("number of dimensions mismatch: %d vs %d",
			len(t.Shape), len(other.Shape)))
	}
	ndim := len(t.Shape)
	if dim < 0 || dim >= ndim {
		panic(fmt.Sprintf("invalid dimension %d (total dimensions %d)", dim, ndim))
	}

	for i := 0; i < ndim; i++ {
		if i != dim && t.Shape[i] != other.Shape[i] {
			panic(fmt.Sprintf("dimension %d size mismatch: %d vs %d",
				i, t.Shape[i], other.Shape[i]))
		}
	}

	newShape := make([]int, ndim)
	copy(newShape, t.Shape)
	newShape[dim] = t.Shape[dim] + other.Shape[dim]

	tStrides := computeStrides(t.Shape)
	otherStrides := computeStrides(other.Shape)
	newStrides := computeStrides(newShape)

	newData := make([]float32, product(newShape))

	for pos := 0; pos < len(newData); pos++ {
		indices := make([]int, ndim)
		remaining := pos
		for d := 0; d < ndim; d++ {
			indices[d] = remaining / newStrides[d]
			remaining = remaining % newStrides[d]
		}

		var src *Tensor
		var srcStrides []int
		if indices[dim] < t.Shape[dim] {
			src = t
			srcStrides = tStrides
		} else {
			src = other
			srcStrides = otherStrides
			indices[dim] -= t.Shape[dim]
		}

		srcPos := 0
		for d := 0; d < ndim; d++ {
			srcPos += indices[d] * srcStrides[d]
		}

		if srcPos < len(src.Data) {
			newData[pos] = src.Data[srcPos]
		}
	}

	return &Tensor{
		Data:  newData,
		Shape: newShape,
	}
}

func (t *Tensor) MaxByDim(dim int, keepdim bool) *Tensor {
	if dim < 0 || dim >= len(t.Shape) {
		panic(fmt.Sprintf("invalid dimension %d, tensor dimension is %d", dim, len(t.Shape)))
	}

	outputShape := make([]int, len(t.Shape))
	copy(outputShape, t.Shape)
	if keepdim {
		outputShape[dim] = 1
	} else {
		outputShape = append(outputShape[:dim], outputShape[dim+1:]...)
	}

	elementSize := 1
	for i := dim + 1; i < len(t.Shape); i++ {
		elementSize *= t.Shape[i]
	}

	totalSlices := 1
	for i := 0; i < dim; i++ {
		totalSlices *= t.Shape[i]
	}

	result := make([]float32, totalSlices*elementSize)

	for sliceIdx := 0; sliceIdx < totalSlices; sliceIdx++ {
		start := sliceIdx * t.Shape[dim] * elementSize

		for pos := 0; pos < elementSize; pos++ {
			maxVal := math.Inf(-1)

			for d := 0; d < t.Shape[dim]; d++ {
				idx := start + d*elementSize + pos
				if t.Data[idx] > maxVal {
					maxVal = t.Data[idx]
				}
			}

			result[sliceIdx*elementSize+pos] = maxVal
		}
	}

	return &Tensor{
		Data:  result,
		Shape: outputShape,
	}
}

func (t *Tensor) getIndices(flatIndex int) []int {
	indices := make([]int, len(t.Shape))
	remaining := flatIndex

	for i := len(t.Shape) - 1; i >= 0; i-- {
		dimSize := t.Shape[i]
		indices[i] = remaining % dimSize
		remaining = remaining / dimSize
	}
	return indices
}

func (t *Tensor) getBroadcastedValue(indices []int) float32 {
	mappedIndices := make([]int, len(t.Shape))

	for i := 0; i < len(t.Shape); i++ {
		if i >= len(indices) {
			mappedIndices[i] = 0
			continue
		}

		dimSize := t.Shape[i]

		if dimSize == 1 {
			mappedIndices[i] = 0
		} else {
			if indices[i] >= dimSize {
				return 0
			}
			mappedIndices[i] = indices[i]
		}
	}

	return t.Get(mappedIndices)
}

func (t *Tensor) SumByDim2(dim int, keepdim bool) *Tensor {
	if dim < 0 || dim >= len(t.Shape) {
		panic(fmt.Sprintf("invalid dimension %d", dim))
	}

	outputShape := make([]int, len(t.Shape))
	copy(outputShape, t.Shape)
	if keepdim {
		outputShape[dim] = 1
	} else {
		outputShape = append(outputShape[:dim], outputShape[dim+1:]...)
	}

	elementSize := 1
	for i := dim + 1; i < len(t.Shape); i++ {
		elementSize *= t.Shape[i]
	}

	totalSlices := 1
	for i := 0; i < dim; i++ {
		totalSlices *= t.Shape[i]
	}

	result := make([]float32, totalSlices*elementSize)

	for sliceIdx := 0; sliceIdx < totalSlices; sliceIdx++ {
		start := sliceIdx * t.Shape[dim] * elementSize
		for pos := 0; pos < elementSize; pos++ {
			var sum float32
			for d := 0; d < t.Shape[dim]; d++ {
				idx := start + d*elementSize + pos
				sum += t.Data[idx]
			}
			result[sliceIdx*elementSize+pos] = sum
		}
	}

	return &Tensor{
		Data:  result,
		Shape: outputShape,
	}
}

func (t *Tensor) MaskedFill(mask *Tensor, value float32) *Tensor {
	if !canBroadcast(t.Shape, mask.Shape) {
		panic("mask shape is not broadcastable")
	}

	outData := make([]float32, len(t.Data))
	copy(outData, t.Data)

	for i := range outData {
		idx := t.getIndices(i)
		maskIndices := make([]int, len(mask.Shape))
		startIdx := len(idx) - len(mask.Shape)
		if startIdx < 0 {
			panic("mask dimensions exceed tensor dimensions")
		}
		for dim := 0; dim < len(mask.Shape); dim++ {
			originalDimIdx := idx[startIdx+dim]
			if mask.Shape[dim] == 1 {
				maskIndices[dim] = 0
			} else {
				maskIndices[dim] = originalDimIdx
			}
		}
		maskVal := mask.GetValue(maskIndices)
		if maskVal != 0 {
			outData[i] = value
		}
	}
	return &Tensor{
		Data:  outData,
		Shape: t.Shape,
	}
}

func (t *Tensor) GetValue(indices []int) float32 {
	if len(indices) != len(t.Shape) {
		panic("indices length does not match tensor dimensions")
	}
	flat := 0
	stride := 1
	for i := len(indices) - 1; i >= 0; i-- {
		if indices[i] >= t.Shape[i] || indices[i] < 0 {
			panic("index out of range")
		}
		flat += indices[i] * stride
		stride *= t.Shape[i]
	}
	return t.Data[flat]
}

func (t *Tensor) SoftmaxByDim(dim int) *Tensor {
	maxVals := t.MaxByDim(dim, true)
	expandedMax := maxVals.Expand(t.Shape)

	shifted := t.Sub(expandedMax)

	expData := shifted.Apply(func(x float32) float32 {
		return math.Exp(x)
	})

	sumExp := expData.SumByDim2(dim, true)
	normalized := expData.Div(sumExp.Expand(t.Shape))

	return normalized
}

func (m *Tensor) ShapeCopy() []int {
	if m.Shape == nil {
		return nil
	}

	copyShape := make([]int, len(m.Shape))
	copy(copyShape, m.Shape)

	return copyShape
}

func (t *Tensor) RepeatInterleave(dim int, repeats int) *Tensor {
	if len(t.Shape) != 2 && len(t.Shape) != 4 {
		panic("RepeatInterleave currently only supports 2D or 4D tensors")
	}

	var newData []float32
	var newShape []int

	if len(t.Shape) == 2 {
		rows, cols := t.Shape[0], t.Shape[1]
		switch dim {
		case 0:
			newData = make([]float32, rows*repeats*cols)
			newShape = []int{rows * repeats, cols}
			for i := 0; i < rows; i++ {
				for r := 0; r < repeats; r++ {
					copy(newData[(i*repeats+r)*cols:(i*repeats+r+1)*cols],
						t.Data[i*cols:(i+1)*cols])
				}
			}
		case 1:
			newData = make([]float32, rows*cols*repeats)
			newShape = []int{rows, cols * repeats}
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					for r := 0; r < repeats; r++ {
						newData[i*cols*repeats+j*repeats+r] = t.Data[i*cols+j]
					}
				}
			}
		default:
			panic("Invalid dimension for 2D tensor")
		}
	} else {
		batch, channels, height, width := t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]
		elementSize := height * width
		switch dim {
		case 0:
			newData = make([]float32, batch*repeats*channels*elementSize)
			newShape = []int{batch * repeats, channels, height, width}
			for i := 0; i < batch; i++ {
				for r := 0; r < repeats; r++ {
					copy(newData[(i*repeats+r)*channels*elementSize:(i*repeats+r+1)*channels*elementSize],
						t.Data[i*channels*elementSize:(i+1)*channels*elementSize])
				}
			}
		case 1:
			newData = make([]float32, batch*channels*repeats*elementSize)
			newShape = []int{batch, channels * repeats, height, width}
			for b := 0; b < batch; b++ {
				for c := 0; c < channels; c++ {
					for r := 0; r < repeats; r++ {
						srcStart := b*channels*elementSize + c*elementSize
						destStart := b*channels*repeats*elementSize + (c*repeats+r)*elementSize
						copy(newData[destStart:destStart+elementSize],
							t.Data[srcStart:srcStart+elementSize])
					}
				}
			}
		default:
			panic("Spatial dimension repeating not implemented")
		}
	}
	return NewTensor(newData, newShape)
}

func (t *Tensor) RoundTo(decimals int) *Tensor {
	if decimals < 0 {
		decimals = 0
	} else if decimals > 8 {
		decimals = 8
	}

	scale := math.Pow10(decimals)
	newData := make([]float32, len(t.Data))
	for i, v := range t.Data {
		scaledValue := float32(v) * scale
		rounded := math.Round(scaledValue)
		newData[i] = float32(rounded / scale)
	}
	return NewTensor(newData, t.Shape)
}
