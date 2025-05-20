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
	github.com/stretchr/testify v1.8.0
	github.com/viterin/vek v0.4.2
	github.com/x448/float16 v0.8.4
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/viterin/partial v1.1.0 // indirect
	golang.org/x/exp v0.0.0-20230817173708-d852ddb80c63 // indirect
	golang.org/x/sys v0.11.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace (
	github.com/chewxy/math32 v1.11.1 => ./thirdparty/math32
	gpu_hardware v0.0.0 => ./restricted/pkg/gpu
)
