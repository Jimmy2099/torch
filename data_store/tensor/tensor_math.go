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
			emptyShape := getBroadcastedShape(t.shape, other.shape)
			return &Tensor{
				shape: emptyShape,
				Data:  make([]float32, 0),
			}
		}
		if !canBroadcast(t.shape, other.shape) {
			panic(fmt.Sprintf("cannot broadcast shapes %v and %v", t.shape, other.shape))
		}
	}

	{
		broadcastedShape := getBroadcastedShape(t.shape, other.shape)
		result := Zeros(broadcastedShape)

		tStrides := computeStrides(t.shape)
		otherStrides := computeStrides(other.shape)

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
			emptyShape := getBroadcastedShape(t.shape, other.shape)
			return &Tensor{
				shape: emptyShape,
				Data:  make([]float32, 0),
			}
		}
		if !canBroadcast(t.shape, other.shape) {
			panic(fmt.Sprintf("cannot broadcast shapes %v and %v", t.shape, other.shape))
		}
	}

	{
		broadcastedShape := getBroadcastedShape(t.shape, other.shape)
		result := Zeros(broadcastedShape)

		tStrides := computeStrides(t.shape)
		otherStrides := computeStrides(other.shape)

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
			emptyShape := getBroadcastedShape(t.shape, other.shape)
			return &Tensor{
				shape: emptyShape,
				Data:  make([]float32, 0),
			}
		}
		if !canBroadcast(t.shape, other.shape) {
			panic(fmt.Sprintf("cannot broadcast shapes %v and %v", t.shape, other.shape))
		}
	}

	broadcastedShape := getBroadcastedShape(t.shape, other.shape)
	result := Zeros(broadcastedShape)

	tStrides := computeStrides(t.shape)
	otherStrides := computeStrides(other.shape)

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
			emptyShape := getBroadcastedShape(t.shape, other.shape)
			return &Tensor{
				shape: emptyShape,
				Data:  make([]float32, 0),
			}
		}
		if !canBroadcast(t.shape, other.shape) {
			panic(fmt.Sprintf("cannot broadcast shapes %v and %v", t.shape, other.shape))
		}
	}
	{

		broadcastedShape := getBroadcastedShape(t.shape, other.shape)
		result := Zeros(broadcastedShape)

		tStrides := computeStrides(t.shape)
		otherStrides := computeStrides(other.shape)

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

	if len(a.shape) == 1 {
		a = a.Reshape(append([]int{1}, a.shape...))
		aIs1D = true
	}
	if len(b.shape) == 1 {
		b = b.Reshape(append(b.shape, 1))
		bIs1D = true
	}

	if len(a.Data) == 0 || len(b.Data) == 0 {
		aBatch := a.shape[:len(a.shape)-2]
		bBatch := b.shape[:len(b.shape)-2]
		batchShape := getBroadcastedShape(aBatch, bBatch)
		m := a.shape[len(a.shape)-2]
		p := b.shape[len(b.shape)-1]
		resultShape := append(append([]int{}, batchShape...), m, p)
		if aIs1D {
			resultShape = resultShape[:len(resultShape)-1]
		}
		if bIs1D {
			resultShape = resultShape[:len(resultShape)-1]
		}
		return &Tensor{
			shape: resultShape,
			Data:  make([]float32, 0),
		}
	}

	aLastDim := a.shape[len(a.shape)-1]
	bSecondLastDim := b.shape[len(b.shape)-2]
	if aLastDim != bSecondLastDim {
		panic(fmt.Sprintf("matmul: dimension mismatch (%d vs %d)", aLastDim, bSecondLastDim))
	}

	aBatchShape := a.shape[:len(a.shape)-2]
	bBatchShape := b.shape[:len(b.shape)-2]
	batchShape := getBroadcastedShape(aBatchShape, bBatchShape)
	if batchShape == nil {
		panic(fmt.Sprintf("matmul: cannot broadcast batch shapes %v and %v", aBatchShape, bBatchShape))
	}

	m := a.shape[len(a.shape)-2]
	p := b.shape[len(b.shape)-1]
	resultShape := append(append([]int{}, batchShape...), m, p)
	result := Zeros(resultShape)

	aStrides := computeStrides(a.shape)
	bStrides := computeStrides(b.shape)
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
		newShape := append(result.shape[:len(result.shape)-2], result.shape[len(result.shape)-1])
		result = result.Reshape(newShape)
	}
	if bIs1D {
		result = result.Reshape(result.shape[:len(result.shape)-1])
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
