module github.com/Jimmy2099/torch

go 1.23.7

require (
	github.com/chewxy/math32 v1.11.1
	github.com/tiktoken-go/tokenizer v0.6.1
)

require (
	github.com/dlclark/regexp2 v1.11.5
	github.com/mohanson/rv64 v0.0.0-20230507001935-dafa85b4fd47
	github.com/shogo82148/float16 v0.5.1
	github.com/shogo82148/int128 v0.2.1
	github.com/x448/float16 v0.8.4
)

replace (
	github.com/chewxy/math32 v1.11.1 => ./thirdparty/math32
//gpu_hardware v0.0.0 => ./restricted/pkg/gpu
)
