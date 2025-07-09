module github.com/Jimmy2099/torch

go 1.24.0

toolchain go1.24.2

require (
	github.com/chewxy/math32 v1.11.1
	github.com/tiktoken-go/tokenizer v0.6.1
)

require (
	github.com/dlclark/regexp2 v1.11.5
	github.com/mohanson/rv64 v0.0.0-20230507001935-dafa85b4fd47
	github.com/owulveryck/onnx-go v0.5.0
	github.com/shogo82148/float16 v0.5.1
	github.com/shogo82148/int128 v0.2.1
	github.com/x448/float16 v0.8.4
	google.golang.org/protobuf v1.36.6
)

require (
	github.com/chewxy/hm v1.0.0 // indirect
	github.com/gogo/protobuf v1.3.0 // indirect
	github.com/golang/protobuf v1.5.0 // indirect
	github.com/google/flatbuffers v1.11.0 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/xtgo/set v1.0.0 // indirect
	gonum.org/v1/gonum v0.0.0-20190902003836-43865b531bee // indirect
	gorgonia.org/tensor v0.9.3 // indirect
	gorgonia.org/vecf32 v0.9.0 // indirect
	gorgonia.org/vecf64 v0.9.0 // indirect
)

replace (
	github.com/chewxy/math32 v1.11.1 => ./thirdparty/math32
//gpu_hardware v0.0.0 => ./restricted/pkg/gpu
)
