package matrix

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
)

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

// GetRows 获取指定行的子矩阵
func (m *Matrix) GetRows(start, end int) *Matrix {
	if start < 0 || end > m.Rows || start >= end {
		panic("invalid row range")
	}

	result := NewMatrix(end-start, m.Cols)
	for i := start; i < end; i++ {
		copy(result.Data[i-start], m.Data[i])
	}
	return result
}

// ReorderRows 根据索引重新排列行
func (m *Matrix) ReorderRows(indices []int) *Matrix {
	if len(indices) != m.Rows {
		panic("indices length must match matrix rows")
	}

	result := NewMatrix(m.Rows, m.Cols)
	for i, idx := range indices {
		copy(result.Data[i], m.Data[idx])
	}
	return result
}

// Reshape changes the dimensions of the matrix while maintaining the total number of elements
func (m *Matrix) Reshape(rows, cols int) *Matrix {
	if rows*cols != m.Rows*m.Cols {
		panic(fmt.Sprintf("Cannot reshape matrix of size (%d,%d) to (%d,%d): total elements must match",
			m.Rows, m.Cols, rows, cols))
	}

	result := NewMatrix(rows, cols)

	// Flatten the original matrix
	flatData := make([]float32, 0, m.Rows*m.Cols)
	for i := 0; i < m.Rows; i++ {
		flatData = append(flatData, m.Data[i]...)
	}

	// Fill the new matrix
	idx := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Data[i][j] = flatData[idx]
			idx++
		}
	}

	return result
}

func (m *Matrix) Size() int {
	return m.Cols * m.Rows
}

// At 获取指定位置的元素
func (m *Matrix) At(row, col int) float32 {
	if row < 0 || row >= m.Rows || col < 0 || col >= m.Cols {
		panic("matrix: index out of range")
	}
	return m.Data[row][col]
}
