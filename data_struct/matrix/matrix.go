package matrix

import (
	"fmt"
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

// NewRandomMatrix initializes a matrix with random values
func NewRandomMatrix(rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Data[i][j] = rand.Float64()*2 - 1 // Random between -1 and 1
		}
	}
	return m
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

func NewMatrixFromSlice1D(data []float64, rows, cols int) *Matrix {
	if len(data) != rows*cols {
		panic(fmt.Sprintf("Data length %d does not match rows * cols (%d)", len(data), rows*cols))
	}

	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// Calculate the correct index for the 1D slice
			m.Data[i][j] = data[i*cols+j]
		}
	}
	return m
}
