package matrix

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	"math/rand"
)
type Matrix struct {
	Rows, Cols int
	Data       [][]float32
}

func NewMatrix(rows, cols int) *Matrix {
	m := &Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([][]float32, rows),
	}
	for i := range m.Data {
		m.Data[i] = make([]float32, cols)
	}
	return m
}

func NewRandomMatrix(rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Data[i][j] = rand.Float32()*2 - 1
		}
	}
	return m
}

func NewMatrixFromSlice(data [][]float32) *Matrix {
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

func NewMatrixFromSlice1D(data []float32, rows, cols int) *Matrix {
	if len(data) != rows*cols {
		panic(fmt.Sprintf("Data length %d does not match rows * cols (%d)", len(data), rows*cols))
	}

	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Data[i][j] = data[i*cols+j]
		}
	}
	return m
}
