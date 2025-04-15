package tensor

import "github.com/Jimmy2099/torch/pkg/fmt"

func (t *Tensor) Sub(other *Tensor) *Tensor {
	if false {
		if t.ShapesMatch(other) {
			return t.Sub_bak(other)
		}
	}

	{
		if len(t.Data) == 0 || len(other.Data) == 0 {
			emptyShape := getBroadcastedShape(t.Shape, other.Shape)
			return &Tensor{
				Shape: emptyShape,
				Data:  make([]float32, 0),
			}
		}
		if !canBroadcast(t.Shape, other.Shape) {
			panic(fmt.Sprintf("cannot broadcast shapes %v and %v", t.Shape, other.Shape))
		}
	}

	{
		broadcastedShape := getBroadcastedShape(t.Shape, other.Shape)
		result := Zeros(broadcastedShape)

		tStrides := computeStrides(t.Shape)
		otherStrides := computeStrides(other.Shape)

		size := result.Size()
		for i := 0; i < size; i++ {
			indices := result.Indices(i)
			tIndex := t.broadcastedIndex(indices, tStrides)
			otherIndex := other.broadcastedIndex(indices, otherStrides)
			result.Data[i] = t.Data[tIndex] - other.Data[otherIndex]
		}

		return result
	}
}

func (t *Tensor) Div(other *Tensor) *Tensor {
	if false {
		if t.ShapesMatch(other) {
			return t.Div_bak(other)
		}
	}

	{
		if len(t.Data) == 0 || len(other.Data) == 0 {
			emptyShape := getBroadcastedShape(t.Shape, other.Shape)
			return &Tensor{
				Shape: emptyShape,
				Data:  make([]float32, 0),
			}
		}
		if !canBroadcast(t.Shape, other.Shape) {
			panic(fmt.Sprintf("cannot broadcast shapes %v and %v", t.Shape, other.Shape))
		}
	}

	{
		broadcastedShape := getBroadcastedShape(t.Shape, other.Shape)
		result := Zeros(broadcastedShape)

		tStrides := computeStrides(t.Shape)
		otherStrides := computeStrides(other.Shape)

		size := result.Size()
		for i := 0; i < size; i++ {
			indices := result.Indices(i)
			tIndex := t.broadcastedIndex(indices, tStrides)
			otherIndex := other.broadcastedIndex(indices, otherStrides)
			result.Data[i] = t.Data[tIndex] / other.Data[otherIndex]
		}

		return result
	}
}

func (t *Tensor) Add(other *Tensor) *Tensor {
	{
		if len(t.Data) == 0 || len(other.Data) == 0 {
			emptyShape := getBroadcastedShape(t.Shape, other.Shape)
			return &Tensor{
				Shape: emptyShape,
				Data:  make([]float32, 0),
			}
		}
		if !canBroadcast(t.Shape, other.Shape) {
			panic(fmt.Sprintf("cannot broadcast shapes %v and %v", t.Shape, other.Shape))
		}
	}

	broadcastedShape := getBroadcastedShape(t.Shape, other.Shape)
	result := Zeros(broadcastedShape)

	tStrides := computeStrides(t.Shape)
	otherStrides := computeStrides(other.Shape)

	size := result.Size()
	for i := 0; i < size; i++ {
		indices := result.Indices(i)
		tIndex := t.broadcastedIndex(indices, tStrides)
		otherIndex := other.broadcastedIndex(indices, otherStrides)
		result.Data[i] = t.Data[tIndex] + other.Data[otherIndex]
	}

	return result
}

func (t *Tensor) Mul(other *Tensor) *Tensor {
	{
		if len(t.Data) == 0 || len(other.Data) == 0 {
			emptyShape := getBroadcastedShape(t.Shape, other.Shape)
			return &Tensor{
				Shape: emptyShape,
				Data:  make([]float32, 0),
			}
		}
		if !canBroadcast(t.Shape, other.Shape) {
			panic(fmt.Sprintf("cannot broadcast shapes %v and %v", t.Shape, other.Shape))
		}
	}
	{

		broadcastedShape := getBroadcastedShape(t.Shape, other.Shape)
		result := Zeros(broadcastedShape)

		tStrides := computeStrides(t.Shape)
		otherStrides := computeStrides(other.Shape)

		size := result.Size()
		for i := 0; i < size; i++ {
			indices := result.Indices(i)
			tIndex := t.broadcastedIndex(indices, tStrides)
			otherIndex := other.broadcastedIndex(indices, otherStrides)
			result.Data[i] = t.Data[tIndex] * other.Data[otherIndex]
		}

		return result
	}
}

func (t *Tensor) MatMul(other *Tensor) *Tensor {
	a := t
	b := other
	aIs1D := false
	bIs1D := false

	if len(a.Shape) == 1 {
		a = a.Reshape(append([]int{1}, a.Shape...))
		aIs1D = true
	}
	if len(b.Shape) == 1 {
		b = b.Reshape(append(b.Shape, 1))
		bIs1D = true
	}

	if len(a.Data) == 0 || len(b.Data) == 0 {
		aBatch := a.Shape[:len(a.Shape)-2]
		bBatch := b.Shape[:len(b.Shape)-2]
		batchShape := getBroadcastedShape(aBatch, bBatch)
		m := a.Shape[len(a.Shape)-2]
		p := b.Shape[len(b.Shape)-1]
		resultShape := append(append([]int{}, batchShape...), m, p)
		if aIs1D {
			resultShape = resultShape[:len(resultShape)-1]
		}
		if bIs1D {
			resultShape = resultShape[:len(resultShape)-1]
		}
		return &Tensor{
			Shape: resultShape,
			Data:  make([]float32, 0),
		}
	}

	aLastDim := a.Shape[len(a.Shape)-1]
	bSecondLastDim := b.Shape[len(b.Shape)-2]
	if aLastDim != bSecondLastDim {
		panic(fmt.Sprintf("matmul: dimension mismatch (%d vs %d)", aLastDim, bSecondLastDim))
	}

	aBatchShape := a.Shape[:len(a.Shape)-2]
	bBatchShape := b.Shape[:len(b.Shape)-2]
	batchShape := getBroadcastedShape(aBatchShape, bBatchShape)
	if batchShape == nil {
		panic(fmt.Sprintf("matmul: cannot broadcast batch shapes %v and %v", aBatchShape, bBatchShape))
	}

	m := a.Shape[len(a.Shape)-2]
	p := b.Shape[len(b.Shape)-1]
	resultShape := append(append([]int{}, batchShape...), m, p)
	result := Zeros(resultShape)

	aStrides := computeStrides(a.Shape)
	bStrides := computeStrides(b.Shape)
	batchSize := product(batchShape)
	matrixSize := m * p

	for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
		indices := getBatchIndices(batchIdx, batchShape)
		aBatchIndices := getBroadcastedIndices(indices, aBatchShape, batchShape)
		bBatchIndices := getBroadcastedIndices(indices, bBatchShape, batchShape)

		aOffset := dotProduct(aBatchIndices, aStrides[:len(aBatchIndices)])
		bOffset := dotProduct(bBatchIndices, bStrides[:len(bBatchIndices)])

		aMatrix := a.Data[aOffset : aOffset+m*aLastDim]
		bMatrix := b.Data[bOffset : bOffset+aLastDim*p]

		resultMatrix := matrixMultiply(aMatrix, bMatrix, m, aLastDim, p)

		resultOffset := batchIdx * matrixSize
		copy(result.Data[resultOffset:resultOffset+matrixSize], resultMatrix)
	}

	if aIs1D {
		newShape := append(result.Shape[:len(result.Shape)-2], result.Shape[len(result.Shape)-1])
		result = result.Reshape(newShape)
	}
	if bIs1D {
		result = result.Reshape(result.Shape[:len(result.Shape)-1])
	}

	return result
}

func dotProduct(a, b []int) int {
	sum := 0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func getBatchIndices(batchIdx int, shape []int) []int {
	indices := make([]int, len(shape))
	for i := range shape {
		size := product(shape[i+1:])
		indices[i] = batchIdx / size
		batchIdx %= size
	}
	return indices
}

func getBroadcastedIndices(bcIndices, origShape, bcShape []int) []int {
	offset := len(bcShape) - len(origShape)
	indices := make([]int, len(origShape))
	for i := range origShape {
		bcDim := offset + i
		if origShape[i] == 1 {
			indices[i] = 0
		} else {
			indices[i] = bcIndices[bcDim]
		}
	}
	return indices
}

func matrixMultiply(a, b []float32, m, n, p int) []float32 {
	result := make([]float32, m*p)
	for i := 0; i < m; i++ {
		for k := 0; k < n; k++ {
			aVal := a[i*n+k]
			for j := 0; j < p; j++ {
				result[i*p+j] += aVal * b[k*p+j]
			}
		}
	}
	return result
}
