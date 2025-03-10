package matrix

import (
	"fmt"
	"math/rand"
)

type Matrix struct {
	rows, cols int
	data       [][]float64
}

// NewMatrix creates a new matrix with given dimensions
func NewMatrix(rows, cols int) *Matrix {
	m := &Matrix{
		rows: rows,
		cols: cols,
		data: make([][]float64, rows),
	}
	for i := range m.data {
		m.data[i] = make([]float64, cols)
	}
	return m
}

// RandomizeMatrix initializes a matrix with random values
func RandomizeMatrix(rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.data[i][j] = rand.Float64()*2 - 1 // Random between -1 and 1
		}
	}
	return m
}

// Multiply performs matrix multiplication: a * b
func Multiply(a, b *Matrix) *Matrix {
	if a.cols != b.rows {
		panic(fmt.Sprintf("Matrix dimensions don't match for multiplication: (%d,%d) * (%d,%d)",
			a.rows, a.cols, b.rows, b.cols))
	}

	result := NewMatrix(a.rows, b.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < b.cols; j++ {
			sum := 0.0
			for k := 0; k < a.cols; k++ {
				sum += a.data[i][k] * b.data[k][j]
			}
			result.data[i][j] = sum
		}
	}
	return result
}

// Add performs element-wise addition: a + b
func Add(a, b *Matrix) *Matrix {
	if a.rows != b.rows || a.cols != b.cols {
		panic(fmt.Sprintf("Matrix dimensions don't match for addition: (%d,%d) + (%d,%d)",
			a.rows, a.cols, b.rows, b.cols))
	}

	result := NewMatrix(a.rows, a.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < a.cols; j++ {
			result.data[i][j] = a.data[i][j] + b.data[i][j]
		}
	}
	return result
}

// Subtract performs element-wise subtraction: a - b
func Subtract(a, b *Matrix) *Matrix {
	if a.rows != b.rows || a.cols != b.cols {
		panic(fmt.Sprintf("Matrix dimensions don't match for subtraction: (%d,%d) - (%d,%d)",
			a.rows, a.cols, b.rows, b.cols))
	}

	result := NewMatrix(a.rows, a.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < a.cols; j++ {
			result.data[i][j] = a.data[i][j] - b.data[i][j]
		}
	}
	return result
}

// Transpose returns the transpose of a matrix
func Transpose(m *Matrix) *Matrix {
	result := NewMatrix(m.cols, m.rows)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.data[j][i] = m.data[i][j]
		}
	}
	return result
}
