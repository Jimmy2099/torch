package tensor

import (
	"bytes"
	"encoding/csv"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// Helper function to create a tensor for testing.
// Assumes a NewTensor function exists that validates shape and data.
// If NewTensor is not available, we can manually create the struct,
// but using the constructor is better if it enforces invariants.
func newTestTensor(t *testing.T, shape []int, data []float64) *Tensor {
	// If you have a constructor like NewTensor, use it:
	// tensor, err := NewTensor(shape, data)
	// if err != nil {
	// 	t.Fatalf("Failed to create test tensor: %v", err)
	// }
	// return tensor

	// Manual creation if no constructor available or for specific invalid states
	// Basic validation matching the logic inside SaveToCSV
	expectedSize := 1
	if len(shape) == 0 {
		if len(data) != 1 && len(data) != 0 { // Allow [] shape with 0 or 1 element for testing
			// Let SaveToCSV handle stricter validation if needed for [] shape
		} else {
			expectedSize = len(data) // 0 or 1
		}
	} else {
		for _, dim := range shape {
			if dim < 0 {
				t.Fatalf("Invalid test setup: negative dimension in shape %v", shape)
			}
			expectedSize *= dim
		}
		if len(data) != expectedSize {
			// Allow mismatch for error testing, but SaveToCSV should catch it.
			// t.Logf("Warning: Creating test tensor with shape/data mismatch: shape %v (%d), data len %d", shape, expectedSize, len(data))
		}
	}

	return &Tensor{
		Shape: shape,
		Data:  data,
	}
}

// Helper function to read CSV content into a [][]string
func readCSVFile(t *testing.T, filename string) [][]string {
	t.Helper() // Marks this as a test helper

	file, err := os.Open(filename)
	if err != nil {
		// Handle cases where the file *should* be empty (created but no data written)
		if os.IsNotExist(err) {
			// Check if the test expected an empty file to be created
			// For most tests here, file not existing is an error
			t.Fatalf("CSV file '%s' was not created when expected.", filename)
		}
		// Check if file exists but is empty
		info, statErr := os.Stat(filename)
		if statErr == nil && info.Size() == 0 {
			return [][]string{} // Return empty slice for empty file
		}
		// Otherwise, fail on open error
		t.Fatalf("Failed to open CSV file '%s' for reading: %v", filename, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		// Allow specific CSV errors like EOF on empty file if that's expected
		if err.Error() == "EOF" && len(records) == 0 {
			return [][]string{}
		}
		t.Fatalf("Failed to read CSV data from '%s': %v", filename, err)
	}
	return records
}

// Helper function to compare expected vs actual CSV data (slice of slices of strings)
func compareCSVData(t *testing.T, expected, actual [][]string) {
	t.Helper()
	if !reflect.DeepEqual(expected, actual) {
		// Provide more detailed error message
		var expectedBuf bytes.Buffer
		csv.NewWriter(&expectedBuf).WriteAll(expected) // Error checking omitted for simplicity in test helper output

		var actualBuf bytes.Buffer
		csv.NewWriter(&actualBuf).WriteAll(actual)

		// Or just print the slices directly for easier debugging
		t.Errorf("CSV content mismatch.\nExpected:\n%v\nActual:\n%v", expected, actual)
		// Alternative more visual output:
		// t.Errorf("CSV content mismatch.\n--- Expected ---\n%s\n--- Actual ---\n%s", expectedBuf.String(), actualBuf.String())
	}
}

// --- Test Cases ---

func TestSaveToCSV_SuccessCases(t *testing.T) {
	testCases := []struct {
		name        string
		tensorShape []int
		tensorData  []float64
		expectedCSV [][]string // Expected content as slice of slices of strings
	}{
		{
			name:        "Scalar Tensor",
			tensorShape: []int{},
			tensorData:  []float64{42.5},
			expectedCSV: [][]string{{"42.5"}},
		},
		{
			name:        "1D Vector (Row)",
			tensorShape: []int{4},
			tensorData:  []float64{1.1, 2.2, 3.3, 4.4},
			expectedCSV: [][]string{{"1.1", "2.2", "3.3", "4.4"}},
		},
		{
			name:        "2D Matrix",
			tensorShape: []int{2, 3},
			tensorData:  []float64{1, 2, 3, 4, 5, 6},
			expectedCSV: [][]string{{"1", "2", "3"}, {"4", "5", "6"}},
		},
		{
			name:        "3D Tensor (Flattened)",
			tensorShape: []int{2, 2, 2},
			tensorData:  []float64{1, 2, 3, 4, 5, 6, 7, 8},
			expectedCSV: [][]string{
				{"1", "2"}, // First inner vector of first matrix
				{"3", "4"}, // Second inner vector of first matrix
				{"5", "6"}, // First inner vector of second matrix
				{"7", "8"}, // Second inner vector of second matrix
			},
		},
		{
			name:        "Empty Tensor (Shape [0])",
			tensorShape: []int{0},
			tensorData:  []float64{},
			expectedCSV: [][]string{}, // Expect an empty file (no records)
		},
		{
			name:        "Empty Tensor (Shape [3, 0])",
			tensorShape: []int{3, 0},
			tensorData:  []float64{},
			expectedCSV: [][]string{}, // Expect an empty file (no records)
		},
		{
			name:        "Tensor with Zeros",
			tensorShape: []int{2, 2},
			tensorData:  []float64{1, 0, 0, -5},
			expectedCSV: [][]string{{"1", "0"}, {"0", "-5"}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Arrange
			tensor := newTestTensor(t, tc.tensorShape, tc.tensorData)
			tempDir := t.TempDir() // Create a temporary directory cleaned up automatically
			filename := filepath.Join(tempDir, "output.csv")

			// Act
			err := tensor.SaveToCSV(filename)

			// Assert
			if err != nil {
				t.Fatalf("SaveToCSV failed unexpectedly: %v", err)
			}

			// Verify file content
			actualCSVData := readCSVFile(t, filename)
			compareCSVData(t, tc.expectedCSV, actualCSVData)

			// Verify empty file creation for empty tensors specifically
			if len(tc.tensorData) == 0 {
				info, statErr := os.Stat(filename)
				if statErr != nil {
					t.Fatalf("os.Stat failed for expected empty file %s: %v", filename, statErr)
				}
				if info.Size() != 0 {
					t.Errorf("Expected empty file for empty tensor, but size is %d", info.Size())
				}
			}
		})
	}
}

func TestSaveToCSV_ErrorCases(t *testing.T) {
	tempDir := t.TempDir()
	validFilename := filepath.Join(tempDir, "error_test.csv")
	// Clean up any potential file created during error tests, though SaveToCSV should fail before writing
	defer os.Remove(validFilename)

	testCases := []struct {
		name        string
		tensor      *Tensor
		filename    string
		expectedErr string // Substring of the expected error message
	}{
		{
			name:        "Nil Tensor",
			tensor:      nil,
			filename:    validFilename,
			expectedErr: "cannot save nil tensor",
		},
		{
			name:        "Tensor with Nil Data",
			tensor:      &Tensor{Shape: []int{2}, Data: nil}, // Manually create invalid state
			filename:    validFilename,
			expectedErr: "tensor with nil data",
		},
		{
			name:        "Scalar shape with incorrect data length",
			tensor:      &Tensor{Shape: []int{}, Data: []float64{1, 2}}, // Manually create invalid state
			filename:    validFilename,
			expectedErr: "invalid tensor state: empty shape but 2 data elements", // Matches error in SaveToCSV
		},
		{
			name:        "1D Shape/Data Mismatch (Data too short)",
			tensor:      newTestTensor(t, []int{5}, []float64{1, 2, 3}), // Use helper, mismatch allowed for test
			filename:    validFilename,
			expectedErr: "data length (3) does not match shape[0] (5)",
		},
		{
			name:        "1D Shape/Data Mismatch (Data too long)",
			tensor:      newTestTensor(t, []int{2}, []float64{1, 2, 3}),
			filename:    validFilename,
			expectedErr: "data length (3) does not match shape[0] (2)",
		},
		{
			name:        "ND Shape/Data Mismatch",
			tensor:      newTestTensor(t, []int{2, 2}, []float64{1, 2, 3}), // Expected 4 elements
			filename:    validFilename,
			expectedErr: "data length (3) does not match product of shape dimensions (4)",
		},
		{
			name: "ND Invalid Shape (Last dim zero with data)",
			// This state should ideally be prevented by NewTensor, but test SaveToCSV's defense
			// Note: If data is empty [], it's a valid empty tensor handled earlier.
			// This tests the case where shape implies zero size but data exists.
			tensor:      &Tensor{Shape: []int{2, 0}, Data: []float64{1.0}}, // Manually create inconsistent state
			filename:    validFilename,
			expectedErr: "data length (1) does not match product of shape dimensions (0)", // Caught by general size validation
		},
		{
			name:   "Invalid Filename (causes create error)",
			tensor: newTestTensor(t, []int{1}, []float64{1}),
			// Use a path that likely cannot be created (e.g., empty string or invalid chars depending on OS)
			// Using a directory as a file is a common way to trigger this cross-platform
			filename:    tempDir,                 // Try to write to the directory itself
			expectedErr: "failed to create file", // Error message might vary slightly by OS ("is a directory", "permission denied" etc.)
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Arrange (already done in struct definition)

			// Act
			err := tc.tensor.SaveToCSV(tc.filename)

			// Assert
			if err == nil {
				t.Fatalf("SaveToCSV succeeded unexpectedly, expected error containing '%s'", tc.expectedErr)
			}
			if !strings.Contains(err.Error(), tc.expectedErr) {
				t.Errorf("SaveToCSV returned wrong error.\nExpected substring: %s\nActual error: %v", tc.expectedErr, err)
			}

			// Ensure no file was partially created in error cases (except maybe the invalid filename one)
			if tc.name != "Invalid Filename (causes create error)" {
				if _, statErr := os.Stat(tc.filename); statErr == nil {
					t.Errorf("CSV file '%s' was created unexpectedly during an error condition.", tc.filename)
					os.Remove(tc.filename) // Clean up artifact
				} else if !os.IsNotExist(statErr) {
					t.Logf("Warning: os.Stat failed for '%s' during error check: %v", tc.filename, statErr)
				}
			}
		})
	}
}

func TestSaveAndLoadRoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		data  []float64
	}{
		{"Vector", []int{3}, []float64{1, 2, 3}},
		{"Matrix", []int{2, 3}, []float64{1, 2, 3, 4, 5, 6}},
		{"3D Tensor", []int{2, 2, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		{"Empty Tensor", []int{0}, []float64{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 创建临时文件
			tmpfile, err := os.CreateTemp("", "test.*.csv")
			if err != nil {
				t.Fatalf("Failed to create temp file: %v", err)
			}
			tmpfile.Close()
			defer os.Remove(tmpfile.Name())

			// 创建原始Tensor
			original := &Tensor{
				Data:  tt.data,
				Shape: tt.shape,
			}

			// 保存到CSV
			if err = original.SaveDataAndShapeToCSV(tmpfile.Name()); err != nil {
				t.Fatalf("SaveDataAndShapeToCSV failed: %v", err)
			}

			// 从CSV加载
			loaded, err := LoadFromCSV(tmpfile.Name())
			if err != nil {
				t.Fatalf("LoadDataAndShapeFromCSV failed: %v", err)
			}

			// 验证数据一致性
			if !reflect.DeepEqual(original.Shape, loaded.Shape) {
				t.Errorf("Shape mismatch\nOriginal: %v\nLoaded:   %v", original.Shape, loaded.Shape)
			}

			if !reflect.DeepEqual(original.Data, loaded.Data) {
				t.Errorf("Data mismatch\nOriginal: %v\nLoaded:   %v", original.Data, loaded.Data)
			}
		})
	}
}
