package torch

import (
	"encoding/csv"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

func LoadMatrixFromCSV(filename string) (*tensor.Tensor, error) {
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

	if len(lines) == 0 {
		return nil, fmt.Errorf("empty CSV file")
	}

	rows := len(lines)
	cols := len(lines[0])
	data := make([]float32, 0, rows*cols)

	for i := 0; i < rows; i++ {
		if len(lines[i]) != cols {
			return nil, fmt.Errorf("inconsistent column count in row %d", i)
		}

		for j := 0; j < cols; j++ {
			val, err := strconv.ParseFloat(lines[i][j], 64)
			if err != nil {
				return nil, fmt.Errorf("invalid float value at row %d, col %d: %v", i, j, err)
			}
			data = append(data, float32(val))
		}
	}

	return tensor.NewTensor(data, []int{rows, cols}), nil
}

func LoadImageFromCSV(filename string) *tensor.Tensor {
	t, err := LoadMatrixFromCSV(filename)
	if err != nil {
		panic(err)
	}

	return t.Reshape([]int{1, t.GetShape()[0], t.GetShape()[1]})
}

func LoadFlatDataFromCSV(filePath string) ([]float32, error) {
	file, err := os.Open(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, err
		}
		return nil, fmt.Errorf("failed to open file %s: %w", filePath, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.TrimLeadingSpace = true
	var flatData []float32

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			if parseErr, ok := err.(*csv.ParseError); ok && parseErr.Err == csv.ErrFieldCount {
				fmt.Printf("Warning: CSV field count error in %s, proceeding line-by-line. Error: %v\n", filePath, parseErr)
			} else {
				return nil, fmt.Errorf("failed to read csv record from %s: %w", filePath, err)
			}
		}

		for _, valueStr := range record {
			valueStr = strings.TrimSpace(valueStr)
			if valueStr == "" {
				continue
			}
			value, err := strconv.ParseFloat(valueStr, 64)
			if err != nil {
				return nil, fmt.Errorf("failed to parse float value '%s' in %s: %w", valueStr, filePath, err)
			}
			flatData = append(flatData, float32(value))
		}
	}

	return flatData, nil
}
