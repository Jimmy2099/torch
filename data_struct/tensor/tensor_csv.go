package tensor

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

func (t *Tensor) SaveToCSV(filename string) error {
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

		// Convert float64 slice to string slice for csv writing
		for j, val := range rowData {
			// 'f' format, -1 precision (use necessary digits), 64-bit float
			record[j] = strconv.FormatFloat(val, 'f', -1, 64)
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

func LoadFromCSV(filename string) (tensorResult *Tensor, err error) {
	// --- Open CSV File ---
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %w", filename, err)
	}
	defer file.Close()

	// --- Create CSV Reader and Read All Records ---
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV file %s: %w", filename, err)
	}

	// --- Handle Empty File ---
	if len(records) == 0 {
		return &Tensor{
			Data:  []float64{},
			Shape: []int{}, // 或者根据业务需要定义空 Tensor 的形状
		}, nil
	}

	// --- Validate and Determine Dimensions ---
	numRows := len(records)
	numCols := len(records[0])
	// 确保所有行具有相同的列数
	for i, record := range records {
		if len(record) != numCols {
			return nil, fmt.Errorf("inconsistent number of columns in row %d: expected %d, got %d", i, numCols, len(record))
		}
	}

	// --- Parse Data ---
	data := make([]float64, 0, numRows*numCols)
	for i, record := range records {
		for j, valueStr := range record {
			val, err := strconv.ParseFloat(valueStr, 64)
			if err != nil {
				return nil, fmt.Errorf("failed to parse float value at row %d, column %d: %w", i, j, err)
			}
			data = append(data, val)
		}
	}

	// --- Infer Tensor Shape ---
	var shape []int
	// 如果只有一个元素，则视为标量 (形状为空)
	// 如果只有一行，则视为向量
	// 否则视为二维矩阵
	if numRows == 1 && numCols == 1 {
		shape = []int{}
	} else if numRows == 1 {
		shape = []int{numCols}
	} else {
		shape = []int{numRows, numCols}
	}

	tensorResult = &Tensor{
		Data:  data,
		Shape: shape,
	}

	return tensorResult, nil
}
