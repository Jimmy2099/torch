package testing

import (
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/layer"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/Jimmy2099/torch/pkg/log"
	math "github.com/chewxy/math32"
	"reflect"
	"testing"
)

func TestWeightLayout(t *testing.T) {

}

func TestGetLayerTestResult(t *testing.T) {

	t.Run("linear layer", func(t *testing.T) {
		script := fmt.Sprintf(`torch.nn.Linear(in_features=%d, out_features=%d)`, 64, 64)
		t1 := tensor.Random([]int{64, 64}, -100, 100)
		weights := tensor.Random([]int{64, 64}, -100, 100)
		bias := tensor.Random([]int{64}, -100, 100)

		l := torch.NewLinearLayer(64, 64)
		l.SetWeightsAndShape(weights.Data, weights.shape)
		l.SetBiasAndShape(bias.Data, bias.shape)

		result := GetLayerTestResult(script, l, t1)
		expected := l.Forward(t1)

		if !result.EqualFloat5(expected) {
			t.Errorf("Linear layer failed:\nExpected:\n%v\nGot:\n%v", expected, result)
		}
	})

	t.Run("linear layer tests", func(t *testing.T) {
		testCases := []struct {
			name        string
			inFeatures  int
			outFeatures int
			inputShape  []int
			weightShape []int
			biasShape   []int
		}{
			{
				name:        "64x64 linear layer",
				inFeatures:  64,
				outFeatures: 64,
				inputShape:  []int{64, 64},
				weightShape: []int{64, 64},
				biasShape:   []int{64},
			},
			{
				name:        "128x64 linear layer",
				inFeatures:  128,
				outFeatures: 64,
				inputShape:  []int{32, 128},
				weightShape: []int{64, 128},
				biasShape:   []int{64},
			},
			{
				name:        "256x128 linear layer",
				inFeatures:  256,
				outFeatures: 128,
				inputShape:  []int{16, 256},
				weightShape: []int{128, 256},
				biasShape:   []int{128},
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				script := fmt.Sprintf(
					`torch.nn.Linear(in_features=%d, out_features=%d)`,
					tc.inFeatures,
					tc.outFeatures,
				)

				t1 := tensor.Random(tc.inputShape, -100, 100)
				weights := tensor.Random(tc.weightShape, -100, 100)
				bias := tensor.Random(tc.biasShape, -100, 100)

				l := torch.NewLinearLayer(tc.inFeatures, tc.outFeatures)
				l.SetWeightsAndShape(weights.Data, weights.shape)
				l.SetBiasAndShape(bias.Data, bias.shape)

				result := GetLayerTestResult(script, l, t1)
				t2 := l.Forward(t1)

				if !result.EqualFloat5(t2) {
					t.Errorf("%s failed:\nInput shape: %v\nExpected:\n%v\nGot:\n%v",
						tc.name,
						tc.inputShape,
						t2,
						result)
				}
			})
		}
	})

	t.Run("linear layer bias", func(t *testing.T) {
		inFeatures := 8192
		outFeatures := 64

		script := fmt.Sprintf(
			`torch.nn.Linear(in_features=%d, out_features=%d, bias=True)`,
			inFeatures, outFeatures,
		)
		input := tensor.Random([]int{1, 8192}, -1, 1)
		weights := tensor.Random([]int{outFeatures, inFeatures}, -1, 1)
		bias := tensor.Random([]int{outFeatures}, -1, 1)

		l := torch.NewLinearLayer(inFeatures, outFeatures)
		l.SetWeightsAndShape(weights.Data, weights.shape)
		l.SetBiasAndShape(bias.Data, bias.shape)

		result := GetLayerTestResult(script, l, input)
		expected := l.Forward(input)

		if !result.EqualFloat16(expected) {
			t.Errorf("Linear layer failed:\nExpected:\n%v\nGot:\n%v", expected, result)
		}
	})

	t.Run("conv2d layer", func(t *testing.T) {
		inChannels := 3
		outChannels := 64
		kernelSize := []int{5, 5}
		stride := []int{2, 2}
		padding := []int{2, 2}

		script := fmt.Sprintf(
			`torch.nn.Conv2d(in_channels=%d, out_channels=%d, kernel_size=[%v,%v], stride=[%v,%v], padding=[%v,%v])`,
			inChannels, outChannels, kernelSize[0], kernelSize[1], stride[0], stride[1], padding[0], padding[1],
		)
		input := tensor.Random([]int{1, 3, 64, 64}, -1, 1)
		weights := tensor.Random([]int{64, 3, 5, 5}, -1, 1)
		bias := tensor.Random([]int{64}, -1, 1)

		l := torch.NewConvLayer(inChannels, outChannels, kernelSize[0], stride[0], padding[0])
		l.SetWeightsAndShape(weights.Data, weights.shape)
		l.SetBiasAndShape(bias.Data, bias.shape)

		result := GetLayerTestResult(script, l, input)
		expected := l.Forward(input)

		if !result.EqualFloat32(expected) {
			t.Errorf("Conv2d layer failed:\nExpected:\n%v\nGot:\n%v", expected, result)
		}
	})

	t.Run("conv2d layers", func(t *testing.T) {
		testCases := []struct {
			name        string
			inChannels  int
			outChannels int
			kernelSize  []int
			stride      []int
			padding     []int
			inputShape  []int
			weightShape []int
			biasShape   []int
		}{
			{
				name:        "basic 3x64 conv",
				inChannels:  3,
				outChannels: 64,
				kernelSize:  []int{5, 5},
				stride:      []int{2, 2},
				padding:     []int{2, 2},
				inputShape:  []int{1, 3, 64, 64},
				weightShape: []int{64, 3, 5, 5},
				biasShape:   []int{64},
			},
			{
				name:        "small kernel no padding",
				inChannels:  64,
				outChannels: 128,
				kernelSize:  []int{3, 3},
				stride:      []int{1, 1},
				padding:     []int{0, 0},
				inputShape:  []int{2, 64, 32, 32},
				weightShape: []int{128, 64, 3, 3},
				biasShape:   []int{128},
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				script := fmt.Sprintf(
					`torch.nn.Conv2d(
                    in_channels=%d, 
                    out_channels=%d, 
                    kernel_size=(%d,%d), 
                    stride=(%d,%d), 
                    padding=(%d,%d)
                )`,
					tc.inChannels,
					tc.outChannels,
					tc.kernelSize[0], tc.kernelSize[1],
					tc.stride[0], tc.stride[1],
					tc.padding[0], tc.padding[1],
				)

				input := tensor.Random(tc.inputShape, -1, 1)
				weights := tensor.Random(tc.weightShape, -1, 1)
				bias := tensor.Random(tc.biasShape, -1, 1)

				l := torch.NewConvLayer(
					tc.inChannels,
					tc.outChannels,
					tc.kernelSize[0],
					tc.stride[0],
					tc.padding[0],
				)

				l.SetWeightsAndShape(weights.Data, weights.shape)
				l.SetBiasAndShape(bias.Data, bias.shape)

				result := GetLayerTestResult(script, l, input)
				expected := l.Forward(input)

				if !result.EqualFloat32(expected) {
					t.Errorf("%s failed:\nInput shape: %v\nExpected:\n%v\nGot:\n%v",
						tc.name,
						tc.inputShape,
						expected,
						result)
				}
			})
		}
	})

	t.Run("relu activation", func(t *testing.T) {
		script := `torch.nn.ReLU()`
		input := tensor.Random([]int{64, 64}, -100, 100)

		l := torch.NewReLULayer()
		result := GetLayerTestResult(script, l, input)
		expected := l.Forward(input)

		if !result.EqualFloat32(expected) {
			t.Errorf("ReLU activation failed:\nExpected:\n%v\nGot:\n%v", expected, result)
		}
	})

	t.Run("relu activations", func(t *testing.T) {
		testCases := []struct {
			name       string
			inputShape []int
			minVal     int
			maxVal     int
		}{
			{
				name:       "mixed values 2D",
				inputShape: []int{64, 64},
				minVal:     -100,
				maxVal:     100,
			},
			{
				name:       "all positive",
				inputShape: []int{128},
				minVal:     1,
				maxVal:     100,
			},
			{
				name:       "4D tensor",
				inputShape: []int{16, 256, 32, 32},
				minVal:     -50,
				maxVal:     50,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				script := `torch.nn.ReLU()`
				input := tensor.Random(tc.inputShape, float32(tc.minVal), float32(tc.maxVal))

				l := torch.NewReLULayer()
				result := GetLayerTestResult(script, l, input)
				expected := l.Forward(input)

				if !result.EqualFloat32(expected) {
					t.Errorf("%s failed:\nInput range: [%d,%d]\nGot: %v",
						tc.name,
						tc.minVal,
						tc.maxVal,
						result)
				}
			})
		}
	})

	t.Run("batchnorm2d layer", func(t *testing.T) {
		numFeatures := 64
		eps := float32(1e-5)
		momentum := float32(0.1)

		script := fmt.Sprintf(
			`torch.nn.BatchNorm2d(num_features=%d, eps=%v, momentum=%v).to(dtype=torch.float32)`,
			numFeatures, eps, momentum,
		)
		input := tensor.Random([]int{1, 64, 32, 32}, -1, 1)
		weight := tensor.Random([]int{64}, -1, 1)
		bias := tensor.Random([]int{64}, -1, 1)

		log.Println("golang batchnorm2d layer weight", weight.Data[0])
		log.Println("golang batchnorm2d bias weight", bias.Data[0])

		l := torch.NewBatchNormLayer(numFeatures, eps, momentum)
		l.SetWeightsAndShape(weight.Data, weight.shape)
		l.SetBiasAndShape(bias.Data, bias.shape)

		result := GetLayerTestResult32(script, l, input)
		expected := l.Forward(input)

		if !result.EqualFloat16(expected) {
			t.Errorf("BatchNorm2d failed:\nExpected:\n%v\nGot:\n%v", expected.Data[:5], result.Data[:5])
		}
	})

	t.Run("batchnorm2d layer test", func(t *testing.T) {
		testCases := []struct {
			name        string
			numFeatures int
			eps         float32
			momentum    float32
			inputShape  []int
			weightShape []int
			biasShape   []int
		}{
			{
				name:        "32 features with large eps",
				numFeatures: 32,
				eps:         1e-3,
				momentum:    0.1,
				inputShape:  []int{2, 32, 16, 16},
				weightShape: []int{32},
				biasShape:   []int{32},
			},
			{
				name:        "128 features with high momentum",
				numFeatures: 128,
				eps:         1e-5,
				momentum:    0.5,
				inputShape:  []int{4, 128, 64, 64},
				weightShape: []int{128},
				biasShape:   []int{128},
			},
			{
				name:        "16 features 3D input",
				numFeatures: 16,
				eps:         1e-5,
				momentum:    0.1,
				inputShape:  []int{8, 16, 8, 8},
				weightShape: []int{16},
				biasShape:   []int{16},
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				script := fmt.Sprintf(
					`torch.nn.BatchNorm2d(num_features=%d, eps=%v, momentum=%v).to(dtype=torch.float32)`,
					tc.numFeatures, tc.eps, tc.momentum,
				)
				input := tensor.Random(tc.inputShape, -1, 1)
				weight := tensor.Random(tc.weightShape, -1, 1)
				bias := tensor.Random(tc.biasShape, -1, 1)

				l := torch.NewBatchNormLayer(tc.numFeatures, tc.eps, tc.momentum)
				l.SetWeightsAndShape(weight.Data, weight.shape)
				l.SetBiasAndShape(bias.Data, bias.shape)

				result := GetLayerTestResult32(script, l, input)
				expected := l.Forward(input)

				if !result.EqualFloat16(expected) {
					t.Errorf("%s failed:\nExpected:\n%v\nGot:\n%v",
						tc.name, expected.Data[:5], result.Data[:5])
				}
			})
		}
	})

	t.Run("convtranspose2d layer", func(t *testing.T) {
		inChannels := 512
		outChannels := 256
		kernelSize := []int{5, 5}
		stride := []int{2, 2}
		padding := []int{2, 2}
		outputPadding := []int{1, 1}

		script := fmt.Sprintf(
			`torch.nn.ConvTranspose2d(%d, %d, kernel_size=(%d,%d), stride=(%d,%d), padding=(%d,%d), output_padding=(%d,%d))`,
			inChannels, outChannels, kernelSize[0], kernelSize[1],
			stride[0], stride[1], padding[0], padding[1],
			outputPadding[0], outputPadding[1],
		)
		input := tensor.Random([]int{1, 512, 4, 4}, -1, 1)
		weights := tensor.Random([]int{512, 256, 5, 5}, -1, 1)
		bias := tensor.Random([]int{256}, -1, 1)

		l := layer.NewConvTranspose2dLayer(
			inChannels, outChannels,
			kernelSize[0], kernelSize[1],
			stride[0], stride[1],
			padding[0], padding[1],
			outputPadding[0], outputPadding[1],
		)
		l.SetWeightsAndShape(weights.Data, weights.shape)
		l.SetBiasAndShape(bias.Data, bias.shape)

		result := GetLayerTestResult(script, l, input)
		expected := l.Forward(input)

		if !result.EqualFloat32(expected) {
			t.Errorf("ConvTranspose2d failed:\nExpected:\n%v\nGot:\n%v", expected.Data[:5], result.Data[:5])
		}
	})

	t.Run("convtranspose2d layer test", func(t *testing.T) {
		testCases := []struct {
			name          string
			inChannels    int
			outChannels   int
			kernelSize    []int
			stride        []int
			padding       []int
			outputPadding []int
			inputShape    []int
			weightsShape  []int
			biasShape     []int
		}{
			{
				name:          "256to128_kernel3_stride1",
				inChannels:    256,
				outChannels:   128,
				kernelSize:    []int{3, 3},
				stride:        []int{1, 1},
				padding:       []int{1, 1},
				outputPadding: []int{0, 0},
				inputShape:    []int{1, 256, 8, 8},
				weightsShape:  []int{256, 128, 3, 3},
				biasShape:     []int{128},
			},
			{
				name:          "64to32_kernel4_stride2",
				inChannels:    64,
				outChannels:   32,
				kernelSize:    []int{4, 4},
				stride:        []int{2, 2},
				padding:       []int{1, 1},
				outputPadding: []int{0, 0},
				inputShape:    []int{2, 64, 16, 16},
				weightsShape:  []int{64, 32, 4, 4},
				biasShape:     []int{32},
			},
			{
				name:          "128to64_kernel7_outputpad1",
				inChannels:    128,
				outChannels:   64,
				kernelSize:    []int{7, 7},
				stride:        []int{2, 2},
				padding:       []int{3, 3},
				outputPadding: []int{1, 1},
				inputShape:    []int{1, 128, 32, 32},
				weightsShape:  []int{128, 64, 7, 7},
				biasShape:     []int{64},
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				script := fmt.Sprintf(
					`torch.nn.ConvTranspose2d(%d, %d, kernel_size=(%d,%d), stride=(%d,%d), padding=(%d,%d), output_padding=(%d,%d))`,
					tc.inChannels, tc.outChannels,
					tc.kernelSize[0], tc.kernelSize[1],
					tc.stride[0], tc.stride[1],
					tc.padding[0], tc.padding[1],
					tc.outputPadding[0], tc.outputPadding[1],
				)
				input := tensor.Random(tc.inputShape, -1, 1)
				weights := tensor.Random(tc.weightsShape, -1, 1)
				bias := tensor.Random(tc.biasShape, -1, 1)

				l := layer.NewConvTranspose2dLayer(
					tc.inChannels, tc.outChannels,
					tc.kernelSize[0], tc.kernelSize[1],
					tc.stride[0], tc.stride[1],
					tc.padding[0], tc.padding[1],
					tc.outputPadding[0], tc.outputPadding[1],
				)
				l.SetWeightsAndShape(weights.Data, weights.shape)
				l.SetBiasAndShape(bias.Data, bias.shape)

				result := GetLayerTestResult(script, l, input)
				expected := l.Forward(input)

				if !result.EqualFloat32(expected) {
					t.Errorf("%s failed:\nExpected:\n%v\nGot:\n%v",
						tc.name, expected.shape, result.shape)
				}
			})
		}
	})

	t.Run("flatten layer", func(t *testing.T) {
		script := `torch.nn.Flatten(start_dim=0, end_dim=-1)`
		input := tensor.Random([]int{1, 512, 4, 4}, -1, 1)
		expectedShape := []int{1, 512 * 4 * 4}

		l := torch.NewFlattenLayer()
		result := GetLayerTestResult(script, l, input)
		expected := l.Forward(input)
		if !result.EqualFloat32(input.Reshape(expectedShape)) {
			t.Errorf("Data mismatch:\nExpected:\n%v,%v\nResult:\n%v,%v", expected.Data[:5], expected.shape, result.Data[:5], result.shape)
		}
	})

	t.Run("flatten layer test", func(t *testing.T) {
		testCases := []struct {
			name       string
			inputShape []int
		}{
			{
				name:       "3D flatten",
				inputShape: []int{4, 3, 28},
			},
			{
				name:       "4D flatten middle dimensions",
				inputShape: []int{2, 3, 28, 28},
			},
			{
				name:       "5D complex flatten",
				inputShape: []int{1, 64, 4, 4, 2},
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				script := `torch.nn.Flatten(start_dim=0, end_dim=-1).to(dtype=torch.float32)`
				input := tensor.Random(tc.inputShape, -1, 1)

				l := torch.NewFlattenLayer()
				result := GetLayerTestResult(script, l, input)
				expected := l.Forward(input)
				if !result.EqualFloat32(expected) {
					t.Errorf("Data mismatch:\nExpected:\n%v,%v\nResult:\n%v,%v", expected.Data[:5], expected.shape, result.Data[:5], result.shape)
				}
			})
		}
	})

	t.Run("tanh layer", func(t *testing.T) {
		script := `torch.nn.Tanh()`
		input := tensor.Random([]int{1, 3, 64, 64}, -1, 1)

		l := layer.NewTanhLayer()
		result := GetLayerTestResult(script, l, input)
		expected := l.Forward(input)

		for _, v := range result.Data {
			if v < -1 || v > 1 {
				t.Errorf("Tanh output out of range: %v", v)
			}
		}
		if !result.EqualFloat32(expected) {
			t.Errorf("Tanh failed:\nExpected:\n%v\nGot:\n%v", expected, result)
		}
	})

	t.Run("tanh layer", func(t *testing.T) {
		testCases := []struct {
			name       string
			inputShape []int
			inputRange [2]float32
		}{
			{
				name:       "large batch small features",
				inputShape: []int{16, 3, 8, 8},
				inputRange: [2]float32{-1, 1},
			},
			{
				name:       "extreme values",
				inputShape: []int{1, 1, 5, 5},
				inputRange: [2]float32{-1000, 1000},
			},
			{
				name:       "3D input",
				inputShape: []int{8, 64, 64},
				inputRange: [2]float32{-5, 5},
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				script := `torch.nn.Tanh()`
				input := tensor.Random(tc.inputShape, float32(tc.inputRange[0]*1000), float32(tc.inputRange[1]*1000))

				l := layer.NewTanhLayer()
				result := GetLayerTestResult(script, l, input)
				expected := l.Forward(input)

				for i, v := range result.Data {
					if v < -1 || v > 1 {
						t.Errorf("%s: Output[%d] out of range: %.4f", tc.name, i, v)
					}
					if math.Abs(float32(v-expected.Data[i])) > 1e-5 {
						t.Errorf("%s: Data mismatch at %d: %.4f vs %.4f",
							tc.name, i, v, expected.Data[i])
					}
				}
			})
		}
	})
}

func TestPointerSafety(t *testing.T) {
	x := tensor.NewTensor([]float32{1, 2}, []int{1, 2})
	layer := torch.NewLinearLayer(2, 3)

	cloned := x.Clone()
	cloned.Data[0] = 9
	if x.Data[0] == 9 {
		t.Fatal("Clone is shallow")
	}

	w := []float32{1, 2, 3, 4, 5, 6}
	layer.SetWeights(w)
	w[0] = 9
	if layer.Weights.Data[0] == 9 {
		t.Fatal("Weight sharing detected")
	}
}

func TestReLUPointerSafety(t *testing.T) {
	t.Run("InputIndependence", func(t *testing.T) {
		relu := torch.NewReLULayer()
		input := tensor.NewTensor([]float32{1.0, -2.0, 3.0}, []int{3})
		output := relu.Forward(input.Clone())

		input.Data[0] = -5.0

		if output.Data[0] != 1.0 {
			t.Error("ReLU output modified by input change")
		}
	})

	t.Run("InplaceOperation", func(t *testing.T) {
		relu := torch.NewReLULayer()
		relu.SetInplace(true)
		input := tensor.NewTensor([]float32{1.0, -2.0, 3.0}, []int{3})
		output := relu.Forward(input)
		fmt.Sprint(output.Data[0])
		if input.Data[1] != -2.0 {
			t.Error("Inplace ReLU incorrectly modified input")
		}
	})
}

func TestBatchNormPointerSafety(t *testing.T) {
	bn := torch.NewBatchNormLayer(256, 1e-5, 0.1)

	t.Run("WeightIndependence", func(t *testing.T) {
		weights := make([]float32, 256)
		copy(weights, bn.GetWeights().Data)

		weights[0] = 999.0

		if bn.GetWeights().Data[0] == 999.0 {
			t.Error("BatchNorm weights sharing detected")
		}
	})

	t.Run("RunningStatsIsolation", func(t *testing.T) {
		originalMean := bn.RunningMean.Clone()

		x := tensor.Ones([]int{64, 256, 8, 8})
		bn.Forward(x)

		if reflect.DeepEqual(bn.RunningMean.Data, originalMean.Data) {
			t.Error("Running mean not updated properly")
		}
	})
}

func TestConvLayerSafety(t *testing.T) {
	t.Run("WeightDeepCopy", func(t *testing.T) {
		conv := torch.NewConvLayer(3, 64, 3, 1, 1)
		original := conv.GetWeights().Clone()
		testData := original.Clone()
		testData.Data[0] = 999.0

		conv.SetWeights(testData.Data)

		if !conv.GetWeights().ShapesMatch(original) {
			t.Errorf("Weight shape changed unexpectedly: got %v want %v",
				conv.GetWeights().shape, original.shape)
		}
	})
}

func TestEmbeddingPointerSafety(t *testing.T) {
	emb := torch.NewEmbedding(10000, 512)

	t.Run("WeightSettingSafety", func(t *testing.T) {
		weights := make([]float32, 10000*512)
		copy(weights, emb.GetWeights().Data)
		weights[0] = 999.0

		emb.SetWeightsAndShape(weights, []int{10000, 512})

		if emb.GradWeights.Data[0] != 0 {
			t.Error("Embedding grad weights not reset")
		}
	})

	t.Run("IndexConversionSafety", func(t *testing.T) {
		indices := tensor.NewTensor([]float32{1.5, 2.0}, []int{1, 2})
		defer func() {
			if r := recover(); r == nil {
				t.Error("Non-integer index not detected")
			}
		}()
		emb.Forward(indices)
	})
}

func TestMaxPool2DSafety(t *testing.T) {
	pool := torch.NewMaxPool2DLayer(2, 2, 0)

	t.Run("InputIsolation", func(t *testing.T) {
		input := tensor.NewTensor([]float32{1, 2, 3, 4}, []int{1, 1, 2, 2})
		output := pool.Forward(input.Clone())

		input.Data[0] = 999.0

		if output.Data[0] != 4.0 {
			t.Error("MaxPool output affected by input modification")
		}
	})

	t.Run("GradientPropagation", func(t *testing.T) {
		input := tensor.Ones([]int{1, 3, 32, 32})
		output := pool.Forward(input)
		gradData := make([]float32, len(output.Data))
		for i := range gradData {
			gradData[i] = 0.5
		}
		gradOutput := tensor.NewTensor(gradData, output.shape)

		gradInput := pool.Backward(gradOutput)

		if !reflect.DeepEqual(gradInput.shape, input.shape) {
			t.Error("MaxPool grad shape mismatch")
		}
	})
}

func TestLayerSafety(t *testing.T) {
	layer := layer.NewRMSNorm(3, 2)

	t.Run("ParameterIndependence", func(t *testing.T) {
		if params := layer.Parameters(); len(params) > 0 {
			original := params[0].Clone()
			testData := make([]float32, len(original.Data))
			copy(testData, original.Data)
			testData[0] = 999.0

			layer.SetWeights(testData)

			if reflect.DeepEqual(params[0].Data, testData) {
				t.Error("Parameter data sharing detected")
			}
		}
	})

	t.Run("InputOutputIsolation", func(t *testing.T) {
		input := tensor.NewTensor([]float32{1, 2, 3}, []int{3})
		output := layer.Forward(input.Clone())
		if output == nil {
			t.Fatal("Output is nil")
		}
		input.Data[0] = 999.0
		if output.Data[0] == 999.0 {
			t.Error("Output shares memory with input")
		}
	})
}
