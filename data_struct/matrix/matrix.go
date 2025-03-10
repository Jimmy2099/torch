package matrix

import (
	"fmt"
	"math"
	"math/rand"
)

type Matrix struct {
	Rows, Cols int
	Data       [][]float64
}

// NewMatrix creates a new matrix with given dimensions
func NewMatrix(rows, cols int) *Matrix {
	m := &Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([][]float64, rows),
	}
	for i := range m.Data {
		m.Data[i] = make([]float64, cols)
	}
	return m
}

// RandomizeMatrix initializes a matrix with random values
func NewRandomMatrix(rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Data[i][j] = rand.Float64()*2 - 1 // Random between -1 and 1
		}
	}
	return m
}

// Multiply performs matrix multiplication: a * b
func Multiply(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic(fmt.Sprintf("Matrix dimensions don't match for multiplication: (%d,%d) * (%d,%d)",
			a.Rows, a.Cols, b.Rows, b.Cols))
	}

	result := NewMatrix(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			sum := 0.0
			for k := 0; k < a.Cols; k++ {
				sum += a.Data[i][k] * b.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}
	return result
}

// Multiply performs matrix multiplication: a * b
func (a *Matrix) Multiply(b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic(fmt.Sprintf("Matrix dimensions don't match for multiplication: (%d,%d) * (%d,%d)",
			a.Rows, a.Cols, b.Rows, b.Cols))
	}

	result := NewMatrix(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			sum := 0.0
			for k := 0; k < a.Cols; k++ {
				sum += a.Data[i][k] * b.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}
	return result
}

// Subtract performs element-wise subtraction: a - b
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

// Subtract performs element-wise subtraction: a - b
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

// Transpose returns the transpose of a matrix
func Transpose(m *Matrix) *Matrix {
	result := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}
	return result
}

// Transpose returns the transpose of a matrix
func (m *Matrix) Transpose() *Matrix {
	result := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}
	return result
}

// Copy creates a deep copy of a matrix
func Copy(m *Matrix) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j]
		}
	}
	return result
}

// Power raises each element of the matrix to the given power
func Power(m *Matrix, power float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = math.Pow(m.Data[i][j], power)
		}
	}
	return result
}

// Apply applies a function to each element of the matrix
func (m *Matrix) Apply(fn func(float64) float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = fn(m.Data[i][j])
		}
	}
	return result
}

// Apply applies a function to each element of the matrix
func Apply(m *Matrix, fn func(float64) float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = fn(m.Data[i][j])
		}
	}
	return result
}

// Concatenate 垂直拼接列向量
func Concatenate(matrices []*Matrix) *Matrix {
	if len(matrices) == 0 {
		return NewMatrix(0, 0)
	}

	features := matrices[0].Rows
	samples := len(matrices)

	// 验证所有矩阵特征数一致
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
	// 输入X的维度为(features, samples)
	// 输出维度为(new_features, samples)

	// 计算新特征的数量
	// 对于 degree=3，特征数为：1（截距） + 2（x1, x2） + 2（x1^2, x2^2） + 2（x1^3, x2^3） = 7
	newFeaturesCount := 1 + X.Rows*degree

	// 创建结果矩阵
	result := NewMatrix(newFeaturesCount, X.Cols)

	// 填充结果矩阵
	for s := 0; s < X.Cols; s++ { // 遍历每个样本
		// 截距项
		result.Data[0][s] = 1.0

		// 原始特征和多项式特征
		idx := 1
		for d := 1; d <= degree; d++ {
			for f := 0; f < X.Rows; f++ {
				result.Data[idx][s] = math.Pow(X.Data[f][s], float64(d))
				idx++
			}
		}
	}

	return result
}

// HadamardProduct performs element-wise multiplication
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

// HadamardProduct performs element-wise multiplication
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

// Sum returns the sum of all elements in the matrix
func (m *Matrix) Sum() float64 {
	sum := 0.0
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			sum += m.Data[i][j]
		}
	}
	return sum
}

// Sum returns the sum of all elements in the matrix
func Sum(m *Matrix) float64 {
	sum := 0.0
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			sum += m.Data[i][j]
		}
	}
	return sum
}

// Dot performs matrix multiplication
func (m *Matrix) Dot(other *Matrix) *Matrix {
	if m.Cols != other.Rows {
		panic(fmt.Sprintf("Matrix dimensions don't match for dot product: (%d,%d) * (%d,%d)",
			m.Rows, m.Cols, other.Rows, other.Cols))
	}

	result := NewMatrix(m.Rows, other.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < other.Cols; j++ {
			sum := 0.0
			for k := 0; k < m.Cols; k++ {
				sum += m.Data[i][k] * other.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}
	return result
}

// Sub subtracts another matrix from this matrix
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

// ZeroGrad resets all gradients to zero
func (m *Matrix) ZeroGrad() {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] = 0
		}
	}
}

// Add performs element-wise addition: a + b
func (m *Matrix) Add(b *Matrix) *Matrix {
	return Add(m, b)
}

// add performs element-wise addition: a + b
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

// SumRows returns a column vector containing the sum of each row
func (m *Matrix) SumRows() *Matrix {
	result := NewMatrix(m.Rows, 1)
	for i := 0; i < m.Rows; i++ {
		sum := 0.0
		for j := 0; j < m.Cols; j++ {
			sum += m.Data[i][j]
		}
		result.Data[i][0] = sum
	}
	return result
}

// MulScalar multiplies each element of the matrix by a scalar value
func (m *Matrix) MulScalar(scalar float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] * scalar
		}
	}
	return result
}

// NewMatrixFromSlice creates a new matrix from a 2D slice
func NewMatrixFromSlice(data [][]float64) *Matrix {
	rows := len(data)
	if rows == 0 {
		return NewMatrix(0, 0)
	}
	cols := len(data[0])

	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		if len(data[i]) != cols {
			panic("All rows must have the same length")
		}
		for j := 0; j < cols; j++ {
			m.Data[i][j] = data[i][j]
		}
	}
	return m
}
