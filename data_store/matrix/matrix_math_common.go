package matrix

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
)

func Multiply(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic(fmt.Sprintf("Matrix dimensions don't match for multiplication: (%d,%d) * (%d,%d)",
			a.Rows, a.Cols, b.Rows, b.Cols))
	}

	result := NewMatrix(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			var sum float32
			for k := 0; k < a.Cols; k++ {
				sum += a.Data[i][k] * b.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}
	return result
}

func (a *Matrix) Multiply(b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic(fmt.Sprintf("Matrix dimensions don't match for multiplication: (%d,%d) * (%d,%d)",
			a.Rows, a.Cols, b.Rows, b.Cols))
	}

	result := NewMatrix(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			var sum float32
			for k := 0; k < a.Cols; k++ {
				sum += a.Data[i][k] * b.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}
	return result
}

func Subtract(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("Matrix dimensions don't match for subtraction: (%d,%d) - (%d,%d)",
			a.Rows, a.Cols, b.Rows, b.Cols))
	}

	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			result.Data[i][j] = a.Data[i][j] - b.Data[i][j]
		}
	}
	return result
}

func (a *Matrix) Subtract(b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("Matrix dimensions don't match for subtraction: (%d,%d) - (%d,%d)",
			a.Rows, a.Cols, b.Rows, b.Cols))
	}

	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			result.Data[i][j] = a.Data[i][j] - b.Data[i][j]
		}
	}
	return result
}

func Transpose(m *Matrix) *Matrix {
	result := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}
	return result
}

func (m *Matrix) Transpose() *Matrix {
	result := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}
	return result
}

func (m *Matrix) Apply(fn func(float32) float32) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = fn(m.Data[i][j])
		}
	}
	return result
}

func Apply(m *Matrix, fn func(float32) float32) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = fn(m.Data[i][j])
		}
	}
	return result
}

func (m *Matrix) SubScalar(s float32) *Matrix {
	return m.Apply(func(x float32) float32 { return x - s })
}

func (m *Matrix) DivScalar(s float32) *Matrix {
	return m.Apply(func(x float32) float32 { return x / s })
}

func (m *Matrix) Max() float32 {
	maxN := math.Inf(-1)
	for i := range m.Data {
		for j := range m.Data[i] {
			if m.Data[i][j] > maxN {
				maxN = m.Data[i][j]
			}
		}
	}
	return maxN
}

func (m *Matrix) Mean() float32 {
	return m.Sum() / float32(m.Rows*m.Cols)
}

func Power(m *Matrix, power float32) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = math.Pow(m.Data[i][j], power)
		}
	}
	return result
}

func Concatenate(matrices []*Matrix) *Matrix {
	if len(matrices) == 0 {
		return NewMatrix(0, 0)
	}

	features := matrices[0].Rows
	samples := len(matrices)

	for _, m := range matrices {
		if m.Rows != features {
			panic("All matrices must have same number of features")
		}
		if m.Cols != 1 {
			panic("Each matrix should be a column vector")
		}
	}

	result := NewMatrix(features, samples)
	for s := 0; s < samples; s++ {
		for f := 0; f < features; f++ {
			result.Data[f][s] = matrices[s].Data[f][0]
		}
	}
	return result
}

func PolynomialFeatures(X *Matrix, degree int) *Matrix {

	newFeaturesCount := 1 + X.Rows*degree

	result := NewMatrix(newFeaturesCount, X.Cols)

	for s := 0; s < X.Cols; s++ {
		result.Data[0][s] = 1.0

		idx := 1
		for d := 1; d <= degree; d++ {
			for f := 0; f < X.Rows; f++ {
				result.Data[idx][s] = math.Pow(X.Data[f][s], float32(d))
				idx++
			}
		}
	}

	return result
}

func (a *Matrix) HadamardProduct(b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("Matrix dimensions don't match for Hadamard product: (%d,%d) * (%d,%d)",
			a.Rows, a.Cols, b.Rows, b.Cols))
	}

	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			result.Data[i][j] = a.Data[i][j] * b.Data[i][j]
		}
	}
	return result
}

func HadamardProduct(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("Matrix dimensions don't match for Hadamard product: (%d,%d) * (%d,%d)",
			a.Rows, a.Cols, b.Rows, b.Cols))
	}

	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			result.Data[i][j] = a.Data[i][j] * b.Data[i][j]
		}
	}
	return result
}

func (m *Matrix) Sum() float32 {
	var sum float32
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			sum += m.Data[i][j]
		}
	}
	return sum
}

func Sum(m *Matrix) float32 {
	var sum float32
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			sum += m.Data[i][j]
		}
	}
	return sum
}

func (m *Matrix) Dot(other *Matrix) *Matrix {
	if m.Cols != other.Rows {
		panic(fmt.Sprintf("Matrix dimensions don't match for dot product: (%d,%d) * (%d,%d)",
			m.Rows, m.Cols, other.Rows, other.Cols))
	}

	result := NewMatrix(m.Rows, other.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < other.Cols; j++ {
			var sum float32
			for k := 0; k < m.Cols; k++ {
				sum += m.Data[i][k] * other.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}
	return result
}

func (m *Matrix) Sub(other *Matrix) *Matrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic(fmt.Sprintf("Matrix dimensions don't match for subtraction: (%d,%d) - (%d,%d)",
			m.Rows, m.Cols, other.Rows, other.Cols))
	}

	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] - other.Data[i][j]
		}
	}
	return result
}

func (m *Matrix) ZeroGrad() {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] = 0
		}
	}
}

func (m *Matrix) Add(b *Matrix) *Matrix {
	return Add(m, b)
}

func Add(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf("Matrix dimensions don't match for addition: (%d,%d) + (%d,%d)",
			a.Rows, a.Cols, b.Rows, b.Cols))
	}

	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			result.Data[i][j] = a.Data[i][j] + b.Data[i][j]
		}
	}
	return result
}

func (m *Matrix) SumRows() *Matrix {
	result := NewMatrix(m.Rows, 1)
	for i := 0; i < m.Rows; i++ {
		var sum float32
		for j := 0; j < m.Cols; j++ {
			sum += m.Data[i][j]
		}
		result.Data[i][0] = sum
	}
	return result
}

func (m *Matrix) MulScalar(scalar float32) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] * scalar
		}
	}
	return result
}

func (m *Matrix) Log() *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = math.Log(m.Data[i][j])
		}
	}
	return result
}

func (m *Matrix) MaxPoolWithArgMax(poolSize, stride int) (*Matrix, []int) {
	outRows := (m.Rows-poolSize)/stride + 1
	outCols := (m.Cols-poolSize)/stride + 1

	result := NewMatrix(outRows, outCols)
	argMax := make([]int, outRows*outCols)

	for i := 0; i < outRows; i++ {
		for j := 0; j < outCols; j++ {
			rowStart := i * stride
			colStart := j * stride
			rowEnd := rowStart + poolSize
			colEnd := colStart + poolSize

			maxVal := m.Data[rowStart][colStart]
			maxIdx := rowStart*m.Cols + colStart
			for r := rowStart; r < rowEnd; r++ {
				for c := colStart; c < colEnd; c++ {
					if m.Data[r][c] > maxVal {
						maxVal = m.Data[r][c]
						maxIdx = r*m.Cols + c
					}
				}
			}

			result.Data[i][j] = maxVal
			argMax[i*outCols+j] = maxIdx
		}
	}

	return result, argMax
}

func (m *Matrix) ReLU() *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = math.Max(0, m.Data[i][j])
		}
	}
	return result
}

func (m *Matrix) Softmax() *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	exp := m.Apply(math.Exp)
	rowSums := make([]float32, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			rowSums[i] += exp.Data[i][j]
		}
	}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = exp.Data[i][j] / rowSums[i]
		}
	}
	return result
}

func (m *Matrix) ArgMax() *Matrix {
	result := NewMatrix(m.Rows, 1)
	for i := 0; i < m.Rows; i++ {
		maxIdx := 0
		maxVal := m.Data[i][0]
		for j := 1; j < m.Cols; j++ {
			if m.Data[i][j] > maxVal {
				maxVal = m.Data[i][j]
				maxIdx = j
			}
		}
		result.Data[i][0] = float32(maxIdx)
	}
	return result
}
