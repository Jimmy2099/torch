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
