package tensor

import (
	"encoding/gob"
	"log"
	"os"
)

var logData []*Tensor

func EnableDataTrace() {
	logData = make([]*Tensor, 0)
}

func EndDataTrace() {
	logData = nil
	matchNum = 0
}

func LoadDataTraceLog() {
	file, err := os.Open("output.gob")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	encoder := gob.NewDecoder(file)
	err = encoder.Decode(&logData)
	if err != nil {
		log.Fatal(err)
	}
}

func WriteDataTraceLog() {
	file, err := os.Create("output.gob")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(logData)
	if err != nil {
		log.Fatal(err)
	}
}

func (t *Tensor) TraceLogToggle() {
	logData = append(logData, t.Clone())
}

var matchNum = 0

func (t *Tensor) Match() bool {
	if matchNum+1 > len(logData) {
		return false
	}
	match := logData[matchNum].Equal(t)
	matchNum += 1
	return match
}

func (t *Tensor) MatchPanic() bool {
	data := logData[matchNum]
	ok := data.EqualFloat32WithShape(t)
	matchNum += 1
	if !ok {
		panic("ERROR: MatchPanic")
		//breakpoint here
	}
	return ok
}

func (t *Tensor) ContainPanic() bool {

	data := logData[matchNum]
	ok := data.Contain(t)
	matchNum += 1
	if !ok {
		data = data.Reshape([]int{len(data.Data), 1})
		data.SaveToCSV("data.csv")
		tt := t.Reshape([]int{len(t.Data), 1})
		tt.SaveToCSV("target.csv")
		panic("ERROR: ContainPanic")
		//set a breakpoint here
	}
	return ok
}
