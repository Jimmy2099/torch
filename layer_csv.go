package torch

import (
	"encoding/csv"
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/matrix"
	"os"
	"strconv"
)

// LoadMatrixFromCSV 辅助函数：从CSV文件加载矩阵
func LoadMatrixFromCSV(filename string) (*matrix.Matrix, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("unable to open CSV file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	lines, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV: %v", err)
	}

	// Assuming the CSV file contains numerical data
	rows := len(lines)
	cols := len(lines[0])

	// Create a matrix of the correct size
	data := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		data[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			val, err := strconv.ParseFloat(lines[i][j], 64)
			if err != nil {
				return nil, fmt.Errorf("invalid float value in CSV: %v", err)
			}
			data[i][j] = val
		}
	}

	// Create the matrix from the data and return it
	return matrix.NewMatrixFromSlice(data), nil
}

// LoadMatrixFromCSV 辅助函数：从CSV文件加载矩阵
func LoadImageFromCSV(filename string) (*matrix.Matrix, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("unable to open CSV file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	lines, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV: %v", err)
	}

	// Assuming the CSV file contains numerical data
	rows := len(lines)
	cols := len(lines[0])

	// Create a matrix of the correct size
	data := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		data[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			val, err := strconv.ParseFloat(lines[i][j], 64)
			if err != nil {
				return nil, fmt.Errorf("invalid float value in CSV: %v", err)
			}
			data[i][j] = val
		}
	}

	// Create the matrix from the data and return it
	return matrix.NewMatrixFromSlice(data), nil
}
