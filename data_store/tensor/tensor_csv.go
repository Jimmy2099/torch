package tensor

import (
	"bufio"
	"encoding/csv"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

const (
	shapeHeaderPrefix = "Shape"
)

func (t *Tensor) SaveToCSV(filename string) error {
	if t == nil || t.Data == nil {
		return fmt.Errorf("cannot save nil tensor or tensor with nil data")
	}

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	shapeRecord := make([]string, len(t.Shape)+1)
	shapeRecord[0] = shapeHeaderPrefix
	for i, dim := range t.Shape {
		shapeRecord[i+1] = strconv.Itoa(dim)
	}
	if err := writer.Write(shapeRecord); err != nil {
		return fmt.Errorf("failed to write shape header: %w", err)
	}

	var numCols, numRows int
	if len(t.Shape) > 0 {
		numCols = t.Shape[len(t.Shape)-1]
	} else {
		numCols = 1
	}

	if numCols == 0 && len(t.Data) == 0 {
		numRows = 0
	} else {
		if numCols == 0 {
			return fmt.Errorf("invalid shape: last dimension is zero with non-empty data")
		}
		numRows = len(t.Data) / numCols
		if len(t.Data)%numCols != 0 {
			return fmt.Errorf("data length %d is not divisible by last dimension %d",
				len(t.Data), numCols)
		}
	}

	record := make([]string, numCols)
	for i := 0; i < numRows; i++ {
		start := i * numCols
		end := start + numCols

		if end > len(t.Data) {
			return fmt.Errorf("data index out of range: %d > %d", end, len(t.Data))
		}

		for j, val := range t.Data[start:end] {
			record[j] = strconv.FormatFloat(float64(val), 'f', -1, 64)
		}

		if err := writer.Write(record); err != nil {
			return fmt.Errorf("failed to write data row %d: %w", i, err)
		}
	}

	return nil
}

func (t *Tensor) SaveToCSVWithoutShape(filename string) error {
	if t == nil || t.Data == nil {
		return fmt.Errorf("cannot save nil tensor or tensor with nil data")
	}

	if len(t.Data) == 0 {
		file, err := os.Create(filename)
		if err != nil {
			return fmt.Errorf("failed to create empty file %s: %w", filename, err)
		}
		file.Close()
		return nil
	}

	numRows := 1
	numCols := 1

	if len(t.Shape) == 0 {
		if len(t.Data) == 1 {
			numRows = 1
			numCols = 1
		} else {
			return fmt.Errorf("invalid tensor state: empty shape but %d data elements", len(t.Data))
		}
	} else if len(t.Shape) == 1 {
		numRows = 1
		numCols = t.Shape[0]
		if numCols == 0 && len(t.Data) == 0 {
		} else if len(t.Data) != numCols {
			return fmt.Errorf("data length (%d) does not match shape[0] (%d)", len(t.Data), numCols)
		}
	} else {
		numCols = t.Shape[len(t.Shape)-1]
		if numCols <= 0 {
			return fmt.Errorf("invalid tensor shape: last dimension is non-positive (%d)", numCols)
		}
		expectedSize := 1
		for _, dim := range t.Shape {
			expectedSize *= dim
		}
		if len(t.Data) != expectedSize {
			return fmt.Errorf("data length (%d) does not match product of shape dimensions (%d)", len(t.Data), expectedSize)
		}
		numRows = len(t.Data) / numCols
	}

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", filename, err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	record := make([]string, numCols)

	for i := 0; i < numRows; i++ {
		startIndex := i * numCols
		endIndex := startIndex + numCols

		if endIndex > len(t.Data) {
			return fmt.Errorf("internal calculation error: trying to access data index %d, but length is %d", endIndex-1, len(t.Data))
		}

		rowData := t.Data[startIndex:endIndex]

		for j, val := range rowData {
			record[j] = strconv.FormatFloat(float64(val), 'f', -1, 64)
		}

		err := writer.Write(record)
		if err != nil {
			return fmt.Errorf("failed to write row %d to csv file %s: %w", i, filename, err)
		}
	}

	if err := writer.Error(); err != nil {
		return fmt.Errorf("error during csv writing/flushing for %s: %w", filename, err)
	}

	return nil
}

func LoadFromCSV(filename string) (*Tensor, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	reader := bufio.NewReaderSize(file, 1024*1024)

	headerLine, err := readLargeLine(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}
	headerRecord := splitCSVLine(headerLine)

	var shape []int
	var records [][]string

	if len(headerRecord) > 0 && strings.TrimSpace(headerRecord[0]) == shapeHeaderPrefix {
		for i := 1; i < len(headerRecord); i++ {
			dim, err := strconv.Atoi(strings.TrimSpace(headerRecord[i]))
			if err != nil || dim < 0 {
				return nil, fmt.Errorf("invalid dimension '%s' at position %d", headerRecord[i], i-1)
			}
			shape = append(shape, dim)
		}
	} else {
		records = append(records, headerRecord)
	}

	for {
		line, err := readLargeLine(reader)
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, fmt.Errorf("failed to read data line: %w", err)
		}
		if len(strings.TrimSpace(line)) == 0 {
			continue
		}
		record := splitCSVLine(line)
		records = append(records, record)
	}

	return loadFromRecords(records, shape)
}

func readLargeLine(reader *bufio.Reader) (string, error) {
	var lineBuilder strings.Builder
	for {
		chunk, isPrefix, err := reader.ReadLine()
		if err != nil {
			return "", err
		}
		lineBuilder.Write(chunk)
		if !isPrefix {
			break
		}
	}
	return lineBuilder.String(), nil
}

func splitCSVLine(line string) []string {
	return strings.Split(line, ",")
}

func loadFromRecords(records [][]string, shape []int) (*Tensor, error) {
	if len(records) == 0 {
		return NewTensor([]float32{}, shape), nil
	}

	var data []float32
	for i, record := range records {
		for j, valStr := range record {
			val, err := strconv.ParseFloat(valStr, 64)
			if err != nil {
				return nil, fmt.Errorf("invalid data value '%s' at row %d, col %d", valStr, i, j)
			}
			data = append(data, float32(val))
		}
	}

	if shape != nil {
		expected := 1
		for _, dim := range shape {
			expected *= dim
		}
		if len(data) != expected {
			return nil, fmt.Errorf("shape %v requires %d elements, got %d", shape, expected, len(data))
		}
	}
	return NewTensor(data, shape), nil
}
