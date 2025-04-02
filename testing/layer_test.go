package testing

import (
	"fmt"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"testing"
)

func TestGetLayerTestResult(t *testing.T) {

	t.Run("linear layer", func(t *testing.T) {
		script := fmt.Sprintf(`torch.nn.Linear(in_features=%d, out_features=%d)`, 64, 64)
		t1 := tensor.Random([]int{64, 64}, -100, 100)
		weights := tensor.Random([]int{64, 64}, -100, 100)
		bias := tensor.Random([]int{64}, -100, 100)

		l := torch.NewLinearLayer(64, 64)
		l.SetWeightsAndShape(weights.Data, weights.Shape)
		l.SetBiasAndShape(bias.Data, bias.Shape)

		result := GetLayerTestResult(script, l, t1)
		t2 := l.Forward(t1)

		if !result.EqualFloat32(t2) {
			t.Errorf("Element-wise multiplication failed:\nExpected:\n%v\nGot:\n%v", t2, result)
		}
	})

}
