package tensor

import (
	"encoding/gob"
	"fmt"
	"os"
	"strings"
)

type Tensor struct {
	Data   []float32
	shape  []int
	Device Device

	//Backward
	Grad         []float32
	RequiresGrad bool
	Parents      []*Tensor
	GradFn       func()
	IsLeaf       bool
}

func NewTensor(data []float32, shape []int) *Tensor {
	if shape == nil {
		shape = []int{1, len(data)}
	}

	sum := ShapeSum(shape)

	if len(data) != sum {
		panic(fmt.Sprintln("shape length mismatch", data, sum))
	}

	t := &Tensor{
		Data:         data,
		shape:        shape,
		RequiresGrad: false,
	}
	t.Device = GetDefaultDevice()
	return t
}

func (t *Tensor) EnableGrad() *Tensor {
	t.RequiresGrad = true
	t.Grad = make([]float32, len(t.Data))
	return t
}

func ShapeSum(shape []int) (result int) {
	if len(shape) == 0 {
		return 0
	}
	result = 1
	for i := 0; i < len(shape); i++ {
		if shape[i] == 1 {
			continue
		}
		result *= shape[i]
	}
	return result
}

func (t *Tensor) TensorData() []float32 {
	return t.Data
}

func (t *Tensor) Save(filename string) error {
	type GobTensor struct {
		Data []float32
	}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	tt := GobTensor{Data: t.Data}
	encoder := gob.NewEncoder(file)
	if err = encoder.Encode(tt); err != nil {
		return err
	}
	return nil
}

func LoadTensorFromGobFile(filename string) (*Tensor, error) {
	file, err := os.Open(filename)
	if err != nil {
		if true {
			{
				csvFileName := strings.Replace(filename, ".gob", ".csv", -1)
				var data *Tensor
				data, err = LoadFromCSV(csvFileName)
				if err != nil {
					panic(fmt.Sprintf("Error loading %s: %v", filename, err))
				}
				if err == nil {
					{
						err = data.Save(filename)
						if err != nil {
							os.Remove(filename)
							panic(err)
						}
						err = os.Remove(csvFileName)
						if err != nil {
							panic(err)
						}
					}
				}
				if err != nil {
					panic(err)
				}
				return data, nil

			}
		}
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	t := &Tensor{}
	if err = decoder.Decode(t); err != nil {
		return nil, err
	}
	return t, nil
}

func LoadTensorIntoGobFile(filename string) (*Tensor, error) {
	t, err := LoadTensorFromGobFile(filename)
	if err != nil {
		if true {
			{
				csvFileName := strings.Replace(filename, ".gob", ".csv", -1)
				var data *Tensor
				data, err = LoadFromCSV(csvFileName)
				if err != nil {
					panic(fmt.Sprintf("Error loading %s: %v", filename, err))
				}
				if err == nil {
					{
						err = data.Save(filename)
						if err != nil {
							os.Remove(filename)
							panic(err)
						}
						err = os.Remove(csvFileName)
						if err != nil {
							panic(err)
						}
					}
				}
				if err != nil {
					panic(err)
				}
				return data, nil

			}
		}
		return nil, err
	}
	return t, nil
}

func (t *Tensor) GetShape() []int {
	return t.shape
}

func (t *Tensor) GetShapeByNum(num int) int {
	return t.shape[num]
}

func NewEmptyTensor() *Tensor {
	return &Tensor{
		Data:  make([]float32, 0),
		shape: make([]int, 0),
	}
}

func (t *Tensor) ToDevice(device Device) {
	if device == nil {
		panic("device cannot be nil")
	}
	//TODO
}

func (t *Tensor) GetData() []float32 {
	return t.Data
}
