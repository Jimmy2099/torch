module github.com/Jimmy2099/torch

go 1.24.0

toolchain go1.24.2

require (
	github.com/chewxy/math32 v1.11.1
	github.com/tiktoken-go/tokenizer v0.6.1
)

require (
	github.com/chewxy/hm v1.0.0
	github.com/davecgh/go-spew v1.1.0
	github.com/disintegration/imaging v1.6.0
	github.com/dlclark/regexp2 v1.11.5
	github.com/gogo/protobuf v1.3.0
	github.com/golang/protobuf v1.5.0
	github.com/kelseyhightower/envconfig v1.4.0
	github.com/kr/pretty v0.1.0
	github.com/mohanson/rv64 v0.0.0-20230507001935-dafa85b4fd47
	github.com/nfnt/resize v0.0.0-20180221191011-83c6a9932646
	github.com/owulveryck/onnx-go v0.5.0
	github.com/pkg/errors v0.9.1
	github.com/sanity-io/litter v1.1.0
	github.com/shogo82148/float16 v0.5.1
	github.com/shogo82148/int128 v0.2.1
	github.com/stretchr/testify v1.4.0
	github.com/vincent-petithory/dataurl v0.0.0-20160330182126-9a301d65acbb
	github.com/x448/float16 v0.8.4
	golang.org/x/image v0.0.0-20180708004352-c73c2afc3b81
	golang.org/x/tools v0.0.0-20190606124116-d0a3d012864b
	gonum.org/v1/gonum v0.0.0-20190902003836-43865b531bee
	gorgonia.org/gorgonia v0.9.4
	gorgonia.org/tensor v0.9.3
)

require (
	github.com/awalterschulze/gographviz v0.0.0-20190522210029-fa59802746ab // indirect
	github.com/google/flatbuffers v1.11.0 // indirect
	github.com/kr/text v0.1.0 // indirect
	github.com/leesper/go_rng v0.0.0-20190531154944-a612b043e353 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/xtgo/set v1.0.0 // indirect
	golang.org/x/net v0.0.0-20190611141213-3f473d35a33a // indirect
	google.golang.org/protobuf v1.36.6 // indirect
	gopkg.in/yaml.v2 v2.2.2 // indirect
	gorgonia.org/cu v0.9.0-beta // indirect
	gorgonia.org/dawson v1.1.0 // indirect
	gorgonia.org/vecf32 v0.9.0 // indirect
	gorgonia.org/vecf64 v0.9.0 // indirect
)

replace (
	github.com/chewxy/math32 v1.11.1 => ./thirdparty/math32
//gpu_hardware v0.0.0 => ./restricted/pkg/gpu
)
