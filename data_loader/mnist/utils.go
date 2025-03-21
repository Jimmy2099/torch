package mnist

import (
	"encoding/binary"
	"os"
)

// 读取图像文件
func loadImages(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magic, num, rows, cols int32
	binary.Read(file, binary.BigEndian, &magic)
	binary.Read(file, binary.BigEndian, &num)
	binary.Read(file, binary.BigEndian, &rows)
	binary.Read(file, binary.BigEndian, &cols)

	imageSize := int(rows * cols)
	data := make([][]float64, num)
	for i := 0; i < int(num); i++ {
		data[i] = make([]float64, imageSize)
		for j := 0; j < imageSize; j++ {
			var pixel uint8
			binary.Read(file, binary.BigEndian, &pixel)
			data[i][j] = float64(pixel) / 255.0 // 归一化
		}
	}
	return data, nil
}

// 读取标签文件
func loadLabels(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magic, num int32
	binary.Read(file, binary.BigEndian, &magic)
	binary.Read(file, binary.BigEndian, &num)

	data := make([][]float64, num)
	for i := 0; i < int(num); i++ {
		var label uint8
		binary.Read(file, binary.BigEndian, &label)
		oneHot := make([]float64, 10)
		oneHot[label] = 1.0 // 转换为 One-hot 编码
		data[i] = oneHot
	}
	return data, nil
}
