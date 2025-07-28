package onnx_runtime

//import (
//	onnx "github.com/owulveryck/onnx-go"
//	"github.com/owulveryck/onnx-go/backend/simple"
//	"io/ioutil"
//	"log"
//)
//
//func UnMarshal() {
//	backend := simple.NewSimpleGraph()
//	b, err := ioutil.ReadFile("./model.onnx")
//	if err != nil {
//		log.Fatal(err)
//	}
//	model := onnx.NewModel(backend)
//	err = model.UnmarshalBinary(b)
//	if err != nil {
//		log.Fatal(err)
//	}
//	_ = model
//}
