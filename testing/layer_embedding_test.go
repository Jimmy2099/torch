package testing

import (
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/algorithm"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
	"math/rand"
	"reflect"
	"testing"
)

// 辅助函数：比较浮点数切片是否近似相等
func floatsEqual(a, b []float32, epsilon float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > epsilon {
			return false
		}
	}
	return true
}

func TestNewEmbedding(t *testing.T) {
	vocabSize := 5
	embDim := 3
	emb := torch.NewEmbedding(vocabSize, embDim)

	// 检查权重形状
	if emb.Weights.Shape[0] != vocabSize || emb.Weights.Shape[1] != embDim {
		t.Errorf("Expected weights shape [%d,%d], got %v", vocabSize, embDim, emb.Weights.Shape)
	}

	// 检查梯度形状
	if emb.GradWeights.Shape[0] != vocabSize || emb.GradWeights.Shape[1] != embDim {
		t.Errorf("Expected gradWeights shape [%d,%d], got %v", vocabSize, embDim, emb.GradWeights.Shape)
	}
}

func TestEmbeddingForward(t *testing.T) {
	// 测试用例表格
	tests := []struct {
		name          string
		input         *tensor.Tensor
		shouldPanic   bool
		expected      []float32
		expectedShape []int
	}{
		{
			name:          "valid_2D_input",
			input:         tensor.NewTensor([]float32{0, 1, 2, 0}, []int{2, 2}),
			shouldPanic:   false,
			expected:      []float32{1, 2, 3, 4, 5, 6, 1, 2},
			expectedShape: []int{2, 2, 2},
		},
		{
			name:          "empty_batch",
			input:         tensor.NewTensor([]float32{}, []int{0, 2}), // 空batch
			shouldPanic:   false,
			expected:      []float32{},
			expectedShape: []int{0, 2, 2},
		},
		{
			name:          "max_index",
			input:         tensor.NewTensor([]float32{2}, []int{1, 1}), // 最大合法索引
			shouldPanic:   false,
			expected:      []float32{5, 6},
			expectedShape: []int{1, 1, 2},
		},
		{
			name:        "non_integer_indices",
			input:       tensor.NewTensor([]float32{0.5, 1.0}, []int{1, 2}),
			shouldPanic: true,
		},
		{
			name:        "3D_input",
			input:       tensor.NewTensor([]float32{0, 1}, []int{1, 1, 2}),
			shouldPanic: true,
		},
		{
			name:        "index_out_of_range",
			input:       tensor.NewTensor([]float32{0, 3}, []int{1, 2}),
			shouldPanic: true,
		},
	}
	emb := &torch.Embedding{
		VocabSize: 3,
		EmbDim:    2,
		Weights:   tensor.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}),
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); (r != nil) != tt.shouldPanic {
					t.Errorf("Panic expected: %v, got: %v", tt.shouldPanic, r != nil)
				}
			}()

			output := emb.Forward(tt.input)

			if !tt.shouldPanic {
				// 验证输出形状
				if !reflect.DeepEqual(output.Shape, tt.expectedShape) {
					t.Errorf("Shape mismatch\nexpected: %v\ngot: %v",
						tt.expectedShape, output.Shape)
				}

				// 验证输出数据
				expectedTensor := tensor.NewTensor(tt.expected, tt.expectedShape)
				if !floatsEqual(output.Data, expectedTensor.Data, 1e-6) {
					t.Errorf("Output mismatch\nexpected: %v\ngot: %v",
						expectedTensor.Data, output.Data)
				}
			}
		})
	}
}

func TestEmbeddingBackward(t *testing.T) {
	emb := &torch.Embedding{
		VocabSize:   3,
		EmbDim:      2,
		Weights:     tensor.NewTensor([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}),
		GradWeights: tensor.NewTensor(make([]float32, 6), []int{3, 2}),
		LastIndices: []int{0, 1, 0, 2}, // 对应batch_size=2, seq_len=2
	}

	gradOutput := tensor.NewTensor([]float32{
		0.1, 0.2, // 第一个元素梯度
		0.3, 0.4, // 第二个元素梯度
		0.5, 0.6, // 第三个元素梯度
		0.7, 0.8, // 第四个元素梯度
	}, []int{2, 2, 2})

	t.Run("gradient_accumulation", func(t *testing.T) {
		emb.Backward(gradOutput, 0.1)

		// 预期梯度累加：
		// 索引0: [0.1+0.5, 0.2+0.6] = [0.6, 0.8]
		// 索引1: [0.3, 0.4]
		// 索引2: [0.7, 0.8]
		expectedGrad := []float32{0.6, 0.8, 0.3, 0.4, 0.7, 0.8}
		if !floatsEqual(emb.GradWeights.Data, expectedGrad, 1e-6) {
			t.Errorf("Gradient accumulation failed\nexpected: %v\ngot: %v", expectedGrad, emb.GradWeights.Data)
		}
	})

	t.Run("weight_update", func(t *testing.T) {
		// 初始权重: [1,2,3,4,5,6]
		// 学习率0.1，梯度如上面测试
		expectedWeights := []float32{
			1 - 0.1*0.6, 2 - 0.1*0.8, // 索引0
			3 - 0.1*0.3, 4 - 0.1*0.4, // 索引1
			5 - 0.1*0.7, 6 - 0.1*0.8, // 索引2
		}

		if !floatsEqual(emb.Weights.Data, expectedWeights, 1e-6) {
			t.Errorf("Weight update failed\nexpected: %v\ngot: %v", expectedWeights, emb.Weights.Data)
		}
	})
}

func TestEmbeddingZeroGrad(t *testing.T) {
	emb := &torch.Embedding{
		GradWeights: tensor.NewTensor([]float32{1, 2, 3, 4}, []int{2, 2}),
	}

	emb.ZeroGrad()
	for _, v := range emb.GradWeights.Data {
		if v != 0 {
			t.Errorf("ZeroGrad failed, found non-zero value: %v", emb.GradWeights.Data)
			break
		}
	}
}

func TestEmbeddingSetWeightsAndShape(t *testing.T) {
	emb := &torch.Embedding{VocabSize: 2, EmbDim: 3}

	t.Run("valid_shape", func(t *testing.T) {
		newWeights := []float32{1, 2, 3, 4, 5, 6}
		emb.SetWeightsAndShape(newWeights, []int{2, 3})

		if !floatsEqual(emb.Weights.Data, newWeights, 0) {
			t.Error("Weights not set correctly")
		}
	})

	t.Run("invalid_shape", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for invalid shape")
			}
		}()
		emb.SetWeightsAndShape([]float32{1, 2}, []int{1, 2})
	})
}

func TestEmbeddingParameters(t *testing.T) {
	weights := tensor.NewTensor([]float32{1, 2}, []int{1, 2})
	emb := &torch.Embedding{Weights: weights}

	params := emb.Parameters()
	if len(params) != 1 || params[0] != weights {
		t.Error("Parameters() should return weights tensor")
	}
}

func TestEmbeddingForwardWithPyTorch(t *testing.T) {
	// 参数化测试用例（修正结构体定义）
	testCases := []struct {
		name       string
		vocabSize  int
		embDim     int
		inputShape []int
	}{
		{
			name:       "small_embedding",
			vocabSize:  5,
			embDim:     3,
			inputShape: []int{2, 2},
		},
		{
			name:       "medium_embedding",
			vocabSize:  10,
			embDim:     8,
			inputShape: []int{3, 4},
		},
		{
			name:       "large_embedding",
			vocabSize:  100,
			embDim:     16,
			inputShape: []int{5, 10},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 生成PyTorch脚本模板
			pyScript := fmt.Sprintf(
				`
torch.nn.Embedding(num_embeddings=%d, embedding_dim=%d)
    in1 = in1.long()`,
				tc.vocabSize,
				tc.embDim,
			)

			// 创建测试输入（带有效索引范围）
			inputData := make([]float32, algorithm.Product(tc.inputShape))
			for i := range inputData {
				// 生成有效索引（0 ≤ index < vocabSize）
				inputData[i] = float32(rand.Intn(tc.vocabSize))
			}
			inputTensor := tensor.NewTensor(inputData, tc.inputShape)

			// 创建并配置Embedding层
			emb := torch.NewEmbedding(tc.vocabSize, tc.embDim)

			// 设置与PyTorch一致的随机权重
			weightsData := make([]float32, tc.vocabSize*tc.embDim)
			scale := math.Sqrt(2.0 / float32(tc.embDim))
			for i := range weightsData {
				weightsData[i] = float32(rand.NormFloat64()) * scale
			}
			emb.SetWeightsAndShape(weightsData, []int{tc.vocabSize, tc.embDim})

			// 处理空输入情况
			if algorithm.Product(tc.inputShape) == 0 {
				output := emb.Forward(inputTensor)
				expectedShape := append(tc.inputShape, tc.embDim)
				if !reflect.DeepEqual(output.Shape, expectedShape) {
					t.Errorf("Expected shape %v, got %v", expectedShape, output.Shape)
				}
				if len(output.Data) != 0 {
					t.Errorf("Empty input should produce empty output")
				}
				return
			}

			// 获取结果（添加包名前缀）
			pyResult := GetLayerTestResult(pyScript, emb, inputTensor)
			goResult := emb.Forward(inputTensor)

			// 验证逻辑保持不变...
			const epsilon = 1e-6
			const relativeTol = 1e-4

			if !reflect.DeepEqual(goResult.Shape, pyResult.Shape) {
				t.Errorf("Shape mismatch\nGo: %v\nPyTorch: %v",
					goResult.Shape, pyResult.Shape)
			}

			if len(goResult.Data) != len(pyResult.Data) {
				t.Fatalf("Data length mismatch\nGo: %d\nPyTorch: %d",
					len(goResult.Data), len(pyResult.Data))
			}

			for i := range goResult.Data {
				diff := math.Abs(goResult.Data[i] - pyResult.Data[i])
				avg := 0.5 * (math.Abs(goResult.Data[i]) + math.Abs(pyResult.Data[i]))

				if diff > epsilon && diff/avg > relativeTol {
					t.Errorf("Value mismatch at index %d\nGo: %.6f\nPyTorch: %.6f",
						i, goResult.Data[i], pyResult.Data[i])
					break
				}
			}
		})
	}
}
