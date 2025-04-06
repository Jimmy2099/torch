package tensor

import (
	"bufio"
	"encoding/csv"
	"fmt"
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

	// 写入形状元数据
	shapeRecord := make([]string, len(t.Shape)+1)
	shapeRecord[0] = shapeHeaderPrefix
	for i, dim := range t.Shape {
		shapeRecord[i+1] = strconv.Itoa(dim)
	}
	if err := writer.Write(shapeRecord); err != nil {
		return fmt.Errorf("failed to write shape header: %w", err)
	}

	// 处理数据写入逻辑
	var numCols, numRows int
	if len(t.Shape) > 0 {
		numCols = t.Shape[len(t.Shape)-1]
	} else {
		numCols = 1
	}

	// 处理空张量情况
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

	// Handle potential edge cases like empty data or shape
	if len(t.Data) == 0 {
		// Option 1: Save an empty file
		file, err := os.Create(filename)
		if err != nil {
			return fmt.Errorf("failed to create empty file %s: %w", filename, err)
		}
		file.Close()
		return nil
		// Option 2: Return an error
		// return fmt.Errorf("cannot save tensor with empty data")
	}

	// --- Determine CSV dimensions ---
	numRows := 1
	numCols := 1

	if len(t.Shape) == 0 { // Scalar Tensor (Shape: [])
		if len(t.Data) == 1 {
			numRows = 1
			numCols = 1
		} else {
			// This case should ideally be prevented by NewTensor validation
			return fmt.Errorf("invalid tensor state: empty shape but %d data elements", len(t.Data))
		}
	} else if len(t.Shape) == 1 { // 1D Tensor (Vector)
		numRows = 1
		numCols = t.Shape[0]
		if numCols == 0 && len(t.Data) == 0 { // Shape [0], Data [] is valid empty
			// Already handled by len(t.Data) == 0 check above
		} else if len(t.Data) != numCols {
			return fmt.Errorf("data length (%d) does not match shape[0] (%d)", len(t.Data), numCols)
		}
	} else { // N-D Tensor (N > 1)
		numCols = t.Shape[len(t.Shape)-1] // Size of the last dimension
		if numCols <= 0 {
			return fmt.Errorf("invalid tensor shape: last dimension is non-positive (%d)", numCols)
		}
		// Calculate expected total size again for validation
		expectedSize := 1
		for _, dim := range t.Shape {
			expectedSize *= dim
		}
		if len(t.Data) != expectedSize {
			return fmt.Errorf("data length (%d) does not match product of shape dimensions (%d)", len(t.Data), expectedSize)
		}
		numRows = len(t.Data) / numCols // All other dimensions flattened into rows
	}

	// --- Create and Open File ---
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", filename, err)
	}
	defer file.Close() // Ensure file is closed even if errors occur

	// --- Create CSV Writer ---
	writer := csv.NewWriter(file)
	// Ensure buffered data is written to the file before closing.
	defer writer.Flush() // Flush called after the function returns, but before file.Close()

	// --- Write Data ---
	record := make([]string, numCols) // Pre-allocate slice for efficiency

	for i := 0; i < numRows; i++ {
		// Calculate the slice of t.Data corresponding to the current row
		startIndex := i * numCols
		endIndex := startIndex + numCols // Slice is exclusive at the end

		// Bounds check (should not happen if validation passes, but good for safety)
		if endIndex > len(t.Data) {
			return fmt.Errorf("internal calculation error: trying to access data index %d, but length is %d", endIndex-1, len(t.Data))
		}

		rowData := t.Data[startIndex:endIndex]

		// Convert float32 slice to string slice for csv writing
		for j, val := range rowData {
			// 'f' format, -1 precision (use necessary digits), 64-bit float
			record[j] = strconv.FormatFloat(float64(val), 'f', -1, 64)
		}

		// Write the record (row) to the CSV file
		err := writer.Write(record)
		if err != nil {
			// An error here might be due to I/O problems
			return fmt.Errorf("failed to write row %d to csv file %s: %w", i, filename, err)
		}
	}

	// Check for any errors potentially buffered by the writer (e.g., during Flush)
	if err := writer.Error(); err != nil {
		return fmt.Errorf("error during csv writing/flushing for %s: %w", filename, err)
	}

	return nil // Success
}

// LoadFromCSV support big row
func LoadFromCSV(filename string) (*Tensor, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// 为处理超长行设置较大的缓冲区（例如1MB）
	reader := bufio.NewReaderSize(file, 1024*1024)

	// 1. 读取形状头（假定形状头行不会特别长）
	headerLine, err := readLargeLine(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}
	headerRecord := splitCSVLine(headerLine)

	var shape []int
	var records [][]string

	// 判断是否为形状头（忽略首尾空格）
	if len(headerRecord) > 0 && strings.TrimSpace(headerRecord[0]) == shapeHeaderPrefix {
		// 解析形状维度
		for i := 1; i < len(headerRecord); i++ {
			dim, err := strconv.Atoi(strings.TrimSpace(headerRecord[i]))
			if err != nil || dim < 0 {
				return nil, fmt.Errorf("invalid dimension '%s' at position %d", headerRecord[i], i-1)
			}
			shape = append(shape, dim)
		}
	} else {
		// 非形状头，将当前行作为数据
		records = append(records, headerRecord)
	}

	// 2. 逐行读取剩余数据
	for {
		line, err := readLargeLine(reader)
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, fmt.Errorf("failed to read data line: %w", err)
		}
		// 忽略空行
		if len(strings.TrimSpace(line)) == 0 {
			continue
		}
		record := splitCSVLine(line)
		records = append(records, record)
	}

	return loadFromRecords(records, shape)
}

// readLargeLine 逐块读取一行数据，避免一次性加载整个超长行到内存
func readLargeLine(reader *bufio.Reader) (string, error) {
	var lineBuilder strings.Builder
	for {
		// ReadLine 可能返回 isPrefix==true 表示行还未结束
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

// splitCSVLine
func splitCSVLine(line string) []string {
	return strings.Split(line, ",")
}

func loadFromRecords(records [][]string, shape []int) (*Tensor, error) {
	// --- Handle Empty File ---
	if len(records) == 0 {
		return NewTensor([]float32{}, shape), nil
	}

	// --- 数据解析 ---
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

	// --- 形状验证 ---
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
