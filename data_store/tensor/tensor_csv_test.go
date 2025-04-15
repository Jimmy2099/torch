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
		if len(data) != 1 && len(data) != 0 {
		} else {
			expectedSize = len(data)
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
	t.Helper()

	file, err := os.Open(filename)
	if err != nil {
		if os.IsNotExist(err) {
			t.Fatalf("CSV file '%s' was not created when expected.", filename)
		}
		info, statErr := os.Stat(filename)
		if statErr == nil && info.Size() == 0 {
			return [][]string{}
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
		csv.NewWriter(&expectedBuf).WriteAll(expected)

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
		expectedCSV [][]string
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
				{"1", "2"},
				{"3", "4"},
				{"5", "6"},
				{"7", "8"},
			},
		},
		{
			name:        "Empty Tensor (Shape [0])",
			tensorShape: []int{0},
			tensorData:  []float32{},
			expectedCSV: [][]string{},
		},
		{
			name:        "Empty Tensor (Shape [3, 0])",
			tensorShape: []int{3, 0},
			tensorData:  []float32{},
			expectedCSV: [][]string{},
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
			tempDir := t.TempDir()
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
		expectedErr string
	}{
		{
			name:        "Nil Tensor",
			tensor:      nil,
			filename:    validFilename,
			expectedErr: "cannot save nil tensor",
		},
		{
			name:        "Tensor with Nil Data",
			tensor:      &Tensor{Shape: []int{2}, Data: nil},
			filename:    validFilename,
			expectedErr: "tensor with nil data",
		},
		{
			name:        "Scalar shape with incorrect data length",
			tensor:      &Tensor{Shape: []int{}, Data: []float32{1, 2}},
			filename:    validFilename,
			expectedErr: "invalid tensor state: empty shape but 2 data elements",
		},
		{
			name:        "1D Shape/Data Mismatch (Data too short)",
			tensor:      newTestTensor(t, []int{5}, []float32{1, 2, 3}),
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
			tensor:      newTestTensor(t, []int{2, 2}, []float32{1, 2, 3}),
			filename:    validFilename,
			expectedErr: "data length (3) does not match product of shape dimensions (4)",
		},
		{
			name: "ND Invalid Shape (Last dim zero with data)",
			tensor:      &Tensor{Shape: []int{2, 0}, Data: []float32{1.0}},
			filename:    validFilename,
			expectedErr: "data length (1) does not match product of shape dimensions (0)",
		},
		{
			name:   "Invalid Filename (causes create error)",
			tensor: newTestTensor(t, []int{1}, []float32{1}),
			filename:    tempDir,
			expectedErr: "failed to create file",
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
					os.Remove(tc.filename)
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
