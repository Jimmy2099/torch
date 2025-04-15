package matrix

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
)

// Multiply performs matrix multiplication: a * b
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

// Multiply performs matrix multiplication: a * b
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

// Apply applies a function to each element of the matrix
func (m *Matrix) Apply(fn func(float32) float32) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = fn(m.Data[i][j])
		}
	}
	return result
}

// Apply applies a function to each element of the matrix
func Apply(m *Matrix, fn func(float32) float32) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = fn(m.Data[i][j])
		}
	}
	return result
}

// SubScalar 矩阵每个元素减去标量
func (m *Matrix) SubScalar(s float32) *Matrix {
	return m.Apply(func(x float32) float32 { return x - s })
}

// DivScalar 矩阵每个元素除以标量
func (m *Matrix) DivScalar(s float32) *Matrix {
	return m.Apply(func(x float32) float32 { return x / s })
}

// Max 获取矩阵最大值
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

// Mean 计算矩阵元素的平均值
func (m *Matrix) Mean() float32 {
	return m.Sum() / float32(m.Rows*m.Cols)
}

// Power raises each element of the matrix to the given power
func Power(m *Matrix, power float32) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = math.Pow(m.Data[i][j], power)
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
				result.Data[idx][s] = math.Pow(X.Data[f][s], float32(d))
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
func (m *Matrix) Sum() float32 {
	var sum float32
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			sum += m.Data[i][j]
		}
	}
	return sum
}

// Sum returns the sum of all elements in the matrix
func Sum(m *Matrix) float32 {
	var sum float32
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
			var sum float32
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
		var sum float32
		for j := 0; j < m.Cols; j++ {
			sum += m.Data[i][j]
		}
		result.Data[i][0] = sum
	}
	return result
}

// MulScalar multiplies each element of the matrix by a scalar value
func (m *Matrix) MulScalar(scalar float32) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] * scalar
		}
	}
	return result
}

// Log 计算矩阵中每个元素的自然对数
func (m *Matrix) Log() *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = math.Log(m.Data[i][j])
		}
	}
	return result
}

// MaxPoolWithArgMax 实现最大池化操作，返回池化结果和最大值位置索引
func (m *Matrix) MaxPoolWithArgMax(poolSize, stride int) (*Matrix, []int) {
	// 计算输出矩阵的尺寸
	outRows := (m.Rows-poolSize)/stride + 1
	outCols := (m.Cols-poolSize)/stride + 1

	// 初始化结果矩阵和索引数组
	result := NewMatrix(outRows, outCols)
	argMax := make([]int, outRows*outCols)

	// 执行最大池化
	for i := 0; i < outRows; i++ {
		for j := 0; j < outCols; j++ {
			// 计算当前池化窗口的位置
			rowStart := i * stride
			colStart := j * stride
			rowEnd := rowStart + poolSize
			colEnd := colStart + poolSize

			// 找到窗口中的最大值及其位置
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

			// 保存结果
			result.Data[i][j] = maxVal
			argMax[i*outCols+j] = maxIdx
		}
	}

	return result, argMax
}

// ReLU 实现ReLU激活函数
func (m *Matrix) ReLU() *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = math.Max(0, m.Data[i][j])
		}
	}
	return result
}

// Softmax 实现Softmax函数
func (m *Matrix) Softmax() *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	// 先计算指数
	exp := m.Apply(math.Exp)
	// 计算每行的和
	rowSums := make([]float32, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			rowSums[i] += exp.Data[i][j]
		}
	}
	// 计算softmax
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = exp.Data[i][j] / rowSums[i]
		}
	}
	return result
}

// ArgMax 返回每行最大值的索引矩阵
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
