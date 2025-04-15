package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"runtime"
	"testing"
	"time"
)

type DataType int16

var DataTypeString = "int16"

const (
	iterations = 1_000_000_000
	warmup     = 100_000_000
)

type TestCase struct {
	dtype     string
	testFuncs map[string]func() DataType
}

var globalSink DataType

func Test_Single_Core_Int16(t *testing.T) {
	runtime.GOMAXPROCS(1)

	testCases := []TestCase{
		{
			dtype: DataTypeString,
			testFuncs: map[string]func() DataType{
				"Addition":       testDataTypeAdd,
				"Subtraction":    testDataTypeSub,
				"Multiplication": testDataTypeMul,
				"Division":       testDataTypeDiv,
			},
		},
	}
	result := ""
	temp := fmt.Sprintf("CPU computation performance test for %s (%d operations)\n\n", DataTypeString, iterations)
	result += temp
	fmt.Print(temp)
	temp = fmt.Sprintf("%-10s %-10s %-15s\n", "Data Type", "Operation", "Performance")
	result += temp
	fmt.Print(temp)

	for _, tc := range testCases {
		for opName, fn := range tc.testFuncs {
			warmupRun(fn)
			opsPerSec := benchmark(fn)
			temp = fmt.Sprintf("%-10s %-10s %-15s\n",
				tc.dtype,
				opName,
				formatUnits(opsPerSec, "OPS"),
			)
			fmt.Print(temp)
			result += temp
		}
	}

	ioutil.WriteFile(fmt.Sprintf("../../single_core_%s_result.txt", DataTypeString), []byte(result), 0644)
	fmt.Fprintf(os.Stdout, "\nBenchmark verification value: %v\n", globalSink)
}

func warmupRun(fn func() DataType) {
	for i := 0; i < warmup; i++ {
		_ = i
	}
}

func testDataTypeAdd() DataType {
	var a, b DataType = 12345, 5678
	var result DataType
	for i := 0; i < iterations; i++ {
		result += a + b
	}
	_ = result
	return 0
}

func testDataTypeSub() DataType {
	var a, b DataType = 12345, 5678
	var result DataType
	for i := 0; i < iterations; i++ {
		result += a - b
	}
	_ = result
	return 0
}

func testDataTypeMul() DataType {
	var a, b DataType = 12345, 5678
	var result DataType
	for i := 0; i < iterations; i++ {
		result += a * b
	}
	_ = result
	return 0
}

func testDataTypeDiv() DataType {
	var a, b DataType = 12345, 5678
	var result DataType
	for i := 0; i < iterations; i++ {
		result += a / b
	}
	_ = result
	return 0
}

func benchmark(fn func() DataType) float64 {
	start := time.Now()
	fn()
	elapsed := time.Since(start).Seconds()
	return float64(iterations) / elapsed
}

func formatUnits(value float64, unit string) string {
	units := []struct {
		threshold float64
		symbol    string
	}{
		{1e15, "P"},
		{1e12, "T"},
		{1e9, "G"},
		{1e6, "M"},
		{1e3, "K"},
		{0, ""},
	}

	for _, u := range units {
		if value >= u.threshold {
			if u.threshold > 0 {
				return fmt.Sprintf("%.2f %s%s", value/u.threshold, u.symbol, unit)
			}
			break
		}
	}
	return fmt.Sprintf("%.2f %s", value, unit)
}
