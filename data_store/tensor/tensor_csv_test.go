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

func newTestTensor(t *testing.T, shape []int, data []float32) *Tensor {

	expectedSize := 1
	if len(shape) == 0 {
		if len(data) != 1 && len(data) != 0 { // Allow [] shape with 0 or 1 element for testing
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
		}
	}

	return &Tensor{
		Shape: shape,
		Data:  data,
	}
}

func readCSVFile(t *testing.T, filename string) [][]string {
	t.Helper() // Marks this as a test helper

	file, err := os.Open(filename)
	if err != nil {
		if os.IsNotExist(err) {
			t.Fatalf("CSV file '%s' was not created when expected.", filename)
		}
		info, statErr := os.Stat(filename)
		if statErr == nil && info.Size() == 0 {
			return [][]string{} // Return empty slice for empty file
		}
		t.Fatalf("Failed to open CSV file '%s' for reading: %v", filename, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		if err.Error() == "EOF" && len(records) == 0 {
			return [][]string{}
		}
		t.Fatalf("Failed to read CSV data from '%s': %v", filename, err)
	}
	return records
}

func compareCSVData(t *testing.T, expected, actual [][]string) {
	t.Helper()
	if !reflect.DeepEqual(expected, actual) {
		var expectedBuf bytes.Buffer
		csv.NewWriter(&expectedBuf).WriteAll(expected) // Error checking omitted for simplicity in test helper output

		var actualBuf bytes.Buffer
		csv.NewWriter(&actualBuf).WriteAll(actual)

		t.Errorf("CSV content mismatch.\nExpected:\n%v\nActual:\n%v", expected, actual)
	}
}


func TestSaveToCSV_SuccessCases(t *testing.T) {
	testCases := []struct {
		name        string
		tensorShape []int
		tensorData  []float32
		expectedCSV [][]string // Expected content as slice of slices of strings
	}{
		{
			name:        "Scalar Tensor",
			tensorShape: []int{},
			tensorData:  []float32{42.5},
			expectedCSV: [][]string{{"42.5"}},
		},
		{
			name:        "1D Vector (Row)",
			tensorShape: []int{4},
			tensorData:  []float32{1.1, 2.2, 3.3, 4.4},
			expectedCSV: [][]string{{"1.1", "2.2", "3.3", "4.4"}},
		},
		{
			name:        "2D Matrix",
			tensorShape: []int{2, 3},
			tensorData:  []float32{1, 2, 3, 4, 5, 6},
			expectedCSV: [][]string{{"1", "2", "3"}, {"4", "5", "6"}},
		},
		{
			name:        "3D Tensor (Flattened)",
			tensorShape: []int{2, 2, 2},
			tensorData:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
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
			tensorData:  []float32{},
			expectedCSV: [][]string{}, // Expect an empty file (no records)
		},
		{
			name:        "Empty Tensor (Shape [3, 0])",
			tensorShape: []int{3, 0},
			tensorData:  []float32{},
			expectedCSV: [][]string{}, // Expect an empty file (no records)
		},
		{
			name:        "Tensor with Zeros",
			tensorShape: []int{2, 2},
			tensorData:  []float32{1, 0, 0, -5},
			expectedCSV: [][]string{{"1", "0"}, {"0", "-5"}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tensor := newTestTensor(t, tc.tensorShape, tc.tensorData)
			tempDir := t.TempDir() // Create a temporary directory cleaned up automatically
			filename := filepath.Join(tempDir, "output.csv")

			err := tensor.SaveToCSV(filename)

			if err != nil {
				t.Fatalf("SaveToCSV failed unexpectedly: %v", err)
			}

			actualCSVData := readCSVFile(t, filename)
			compareCSVData(t, tc.expectedCSV, actualCSVData)

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
			tensor:      &Tensor{Shape: []int{}, Data: []float32{1, 2}}, // Manually create invalid state
			filename:    validFilename,
			expectedErr: "invalid tensor state: empty shape but 2 data elements", // Matches error in SaveToCSV
		},
		{
			name:        "1D Shape/Data Mismatch (Data too short)",
			tensor:      newTestTensor(t, []int{5}, []float32{1, 2, 3}), // Use helper, mismatch allowed for test
			filename:    validFilename,
			expectedErr: "data length (3) does not match shape[0] (5)",
		},
		{
			name:        "1D Shape/Data Mismatch (Data too long)",
			tensor:      newTestTensor(t, []int{2}, []float32{1, 2, 3}),
			filename:    validFilename,
			expectedErr: "data length (3) does not match shape[0] (2)",
		},
		{
			name:        "ND Shape/Data Mismatch",
			tensor:      newTestTensor(t, []int{2, 2}, []float32{1, 2, 3}), // Expected 4 elements
			filename:    validFilename,
			expectedErr: "data length (3) does not match product of shape dimensions (4)",
		},
		{
			name: "ND Invalid Shape (Last dim zero with data)",
			tensor:      &Tensor{Shape: []int{2, 0}, Data: []float32{1.0}}, // Manually create inconsistent state
			filename:    validFilename,
			expectedErr: "data length (1) does not match product of shape dimensions (0)", // Caught by general size validation
		},
		{
			name:   "Invalid Filename (causes create error)",
			tensor: newTestTensor(t, []int{1}, []float32{1}),
			filename:    tempDir,                 // Try to write to the directory itself
			expectedErr: "failed to create file", // Error message might vary slightly by OS ("is a directory", "permission denied" etc.)
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {

			err := tc.tensor.SaveToCSV(tc.filename)

			if err == nil {
				t.Fatalf("SaveToCSV succeeded unexpectedly, expected error containing '%s'", tc.expectedErr)
			}
			if !strings.Contains(err.Error(), tc.expectedErr) {
				t.Errorf("SaveToCSV returned wrong error.\nExpected substring: %s\nActual error: %v", tc.expectedErr, err)
			}

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
		data  []float32
	}{
		{"Vector", []int{3}, []float32{1, 2, 3}},
		{"Matrix", []int{2, 3}, []float32{1, 2, 3, 4, 5, 6}},
		{"3D Tensor", []int{2, 2, 2}, []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		{"Empty Tensor", []int{0}, []float32{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpfile, err := os.CreateTemp("", "test.*.csv")
			if err != nil {
				t.Fatalf("Failed to create temp file: %v", err)
			}
			tmpfile.Close()
			defer os.Remove(tmpfile.Name())

			original := &Tensor{
				Data:  tt.data,
				Shape: tt.shape,
			}

			if err = original.SaveToCSV(tmpfile.Name()); err != nil {
				t.Fatalf("SaveDataAndShapeToCSV failed: %v", err)
			}

			loaded, err := LoadFromCSV(tmpfile.Name())
			if err != nil {
				t.Fatalf("LoadDataAndShapeFromCSV failed: %v", err)
			}

			if !reflect.DeepEqual(original.Shape, loaded.Shape) {
				t.Errorf("Shape mismatch\nOriginal: %v\nLoaded:   %v", original.Shape, loaded.Shape)
			}

			if !reflect.DeepEqual(original.Data, loaded.Data) {
				t.Errorf("Data mismatch\nOriginal: %v\nLoaded:   %v", original.Data, loaded.Data)
			}
		})
	}
}
