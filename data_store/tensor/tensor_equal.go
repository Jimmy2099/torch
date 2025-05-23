package tensor

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/Jimmy2099/torch/pkg/log"
	math "github.com/chewxy/math32"
)

func (t *Tensor) EqualFloat5(other *Tensor) bool {
	if t == nil || other == nil {
		return t == nil && other == nil
	}

	if !shapeEqual(t.shape, other.shape) {
		return false
	}

	if len(t.Data) != len(other.Data) {
		return false
	}

	for i := range t.Data {
		if math.Abs(t.Data[i]-other.Data[i]) > 0.3 {
			fmt.Println("---", fmt.Sprintf("%.5f", t.Data[i]), fmt.Sprintf("%.5f", other.Data[i]))
			log.Println(i, t.Data[i], " != ", other.Data[i])
			return false
		}
	}

	return true
}

func (t *Tensor) EqualFloat16(other *Tensor) bool {
	if t == nil || other == nil {
		return t == nil && other == nil
	}

	if !shapeEqual(t.shape, other.shape) {
		return false
	}

	if len(t.Data) != len(other.Data) {
		return false
	}

	for i := range t.Data {
		if fmt.Sprintf("%.10f", t.Data[i]) != fmt.Sprintf("%.10f", other.Data[i]) {
			log.Println(i, t.Data[i], " != ", other.Data[i])
			return false
		}
	}

	return true
}

func (t *Tensor) EqualFloat32WithShape(other *Tensor) bool {
	if t == nil || other == nil {
		return t == nil && other == nil
	}

	if len(t.Data) != len(other.Data) {
		return false
	}

	for i := range t.Data {
		if t.Data[i] != other.Data[i] {
			log.Println(i, t.Data[i], " != ", other.Data[i])
			t.SaveToCSV("target.csv")
			other.SaveToCSV("other.csv")
			return false
		}
	}

	return true
}

func (t *Tensor) EqualFloat32(other *Tensor) bool {
	if t == nil || other == nil {
		return t == nil && other == nil
	}

	if !shapeEqual(t.shape, other.shape) {
		return false
	}

	if len(t.Data) != len(other.Data) {
		return false
	}

	for i := range t.Data {
		if float32(t.Data[i]) != float32(other.Data[i]) {
			log.Println(i, t.Data[i], " != ", other.Data[i])
			return false
		}
	}

	return true
}

func (t *Tensor) Equal(other *Tensor) bool {
	if t == nil || other == nil {
		return t == nil && other == nil
	}

	if !shapeEqual(t.shape, other.shape) {
		return false
	}

	if len(t.Data) != len(other.Data) {
		return false
	}

	for i := range t.Data {
		if t.Data[i] != other.Data[i] {
			log.Println(i, t.Data[i], " != ", other.Data[i])
			return false
		}
	}

	return true
}

func shapeEqual(shape1, shape2 []int) bool {
	if len(shape1) != len(shape2) {
		return false
	}
	for i := range shape1 {
		if shape1[i] != shape2[i] {
			return false
		}
	}
	return true
}

func (t *Tensor) EqualWithTolerance(other *Tensor, epsilon float32) bool {
	if t == nil || other == nil {
		return t == nil && other == nil
	}

	if !shapeEqual(t.shape, other.shape) {
		return false
	}

	for i := range t.Data {
		if math.Abs(t.Data[i]-other.Data[i]) > epsilon {
			return false
		}
	}

	return true
}

func (t *Tensor) Contain(other *Tensor) bool {
	if other == nil {
		return false
	}

	tDataLen := len(t.Data)
	otherDataLen := len(other.Data)

	if tDataLen == otherDataLen {
		return other.EqualFloat32WithShape(t)
	}

	if tDataLen == otherDataLen {
		for i := 0; i < tDataLen; i++ {
			if t.Data[i] != other.Data[i] {
				return false
			}
		}
		return true
	}

	if otherDataLen == 0 {
		return true
	}
	if tDataLen < otherDataLen {
		return false
	}

	for i := 0; i <= tDataLen-otherDataLen; i++ {
		match := true
		for j := 0; j < otherDataLen; j++ {
			if t.Data[i+j] != other.Data[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}
