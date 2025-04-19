package tensor

import (
	"encoding/gob"
	"log"
	"os"
)

var dataTrace *DataTrace = nil

type DataTrace struct {
	matchPointerNum int
	TensorData      []*Tensor
}

func EnableTensorTrace() {
	if dataTrace != nil {
		panic("dataTrace has already been initialized")
	}
	dataTrace = &DataTrace{
		TensorData:      make([]*Tensor, 0),
		matchPointerNum: 0,
	}
}

func EndTensorTrace() {
	if dataTrace == nil {
		panic("dataTrace has not been initialized")
	}
	dataTrace = nil
}

func (m *DataTrace) LoadDataTraceLog() {
	if dataTrace != nil {
		panic("dataTrace has already been initialized")
	}
	file, err := os.Open("output.gob")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	encoder := gob.NewDecoder(file)
	err = encoder.Decode(&m.TensorData)
	if err != nil {
		log.Fatal(err)
	}
}

func (m *DataTrace) WriteDataTraceLog() {
	if dataTrace == nil {
		panic("dataTrace has not been initialized")
	}
	file, err := os.Create("output.gob")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(m.TensorData)
	if err != nil {
		log.Fatal(err)
	}
}

func GetDataTrace() *DataTrace {
	return dataTrace
}

func (m *DataTrace) AppendTensorData(tensor *Tensor) {
	m.TensorData = append(m.TensorData, tensor)
}

func (t *Tensor) TraceLogToggle() {
	GetDataTrace().AppendTensorData(t.Clone())
}

func (t *Tensor) Match() bool {
	trace := GetDataTrace()
	if trace == nil {
		panic("dataTrace is not initialized")
	}
	if trace.matchPointerNum >= len(trace.TensorData) {
		return false
	}
	data := trace.TensorData[trace.matchPointerNum]
	match := data.Equal(t)
	trace.matchPointerNum++
	return match
}

func (t *Tensor) MatchPanic() bool {
	trace := GetDataTrace()
	if trace == nil {
		panic("dataTrace is not initialized")
	}
	if trace.matchPointerNum >= len(trace.TensorData) {
		panic("MatchPanic: no more data in dataTrace")
	}
	data := trace.TensorData[trace.matchPointerNum]
	ok := data.EqualFloat32WithShape(t)
	trace.matchPointerNum++
	if !ok {
		panic("ERROR: MatchPanic")
	}
	return ok
}

func (t *Tensor) ContainPanic() bool {
	trace := GetDataTrace()
	if trace == nil {
		panic("dataTrace is not initialized")
	}
	if trace.matchPointerNum >= len(trace.TensorData) {
		panic("ContainPanic: no more data in dataTrace")
	}
	data := trace.TensorData[trace.matchPointerNum]
	ok := data.Contain(t)
	trace.matchPointerNum++
	if !ok {
		data = data.Reshape([]int{len(data.Data), 1})
		data.SaveToCSV("data.csv")
		tt := t.Reshape([]int{len(t.Data), 1})
		tt.SaveToCSV("target.csv")
		panic("ERROR: ContainPanic")
	}
	return ok
}
