package data_type

import (
	"fmt"
	"testing"
)

func TestNewDataMemoryFloat(t *testing.T) {
	data := NewDataMemoryFloat32()
	fmt.Println(data.String())
}
