package mnist

import (
	"encoding/binary"
	"os"
)

func loadImages(filename string) ([][]float32, error) {
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
	data := make([][]float32, num)
	for i := 0; i < int(num); i++ {
		data[i] = make([]float32, imageSize)
		for j := 0; j < imageSize; j++ {
			var pixel uint8
			binary.Read(file, binary.BigEndian, &pixel)
			data[i][j] = float32(pixel) / 255.0
		}
	}
	return data, nil
}

func loadLabels(filename string) ([][]float32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magic, num int32
	binary.Read(file, binary.BigEndian, &magic)
	binary.Read(file, binary.BigEndian, &num)

	data := make([][]float32, num)
	for i := 0; i < int(num); i++ {
		var label uint8
		binary.Read(file, binary.BigEndian, &label)
		oneHot := make([]float32, 10)
		oneHot[label] = 1.0
		data[i] = oneHot
	}
	return data, nil
}
