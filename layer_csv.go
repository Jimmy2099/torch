package torch

import (
	"encoding/csv"
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"os"
	"strconv"
)

// LoadMatrixFromCSV 从CSV文件加载数据并返回张量
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
	data := make([]float64, 0, rows*cols)

	for i := 0; i < rows; i++ {
		if len(lines[i]) != cols {
			return nil, fmt.Errorf("inconsistent column count in row %d", i)
		}

		for j := 0; j < cols; j++ {
			val, err := strconv.ParseFloat(lines[i][j], 64)
			if err != nil {
				return nil, fmt.Errorf("invalid float value at row %d, col %d: %v", i, j, err)
			}
			data = append(data, val)
		}
	}

	return tensor.NewTensor(data, []int{rows, cols}), nil
}

// LoadImageFromCSV 从CSV文件加载图像数据
func LoadImageFromCSV(filename string) *tensor.Tensor {
	// 对于图像数据，可能需要添加额外的维度（如通道数）
	t, err := LoadMatrixFromCSV(filename)
	if err != nil {
		panic(err)
	}

	// 假设是灰度图像，添加通道维度
	// 形状从 [rows, cols] 变为 [1, rows, cols]
	return t.Reshape([]int{1, t.Shape[0], t.Shape[1]})
}
