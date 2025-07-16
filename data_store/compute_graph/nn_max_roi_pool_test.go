package compute_graph

import (
	"fmt"
	"testing"
)

func TestMaxRoiPool(t *testing.T) {
	graph := NewComputationalGraph()

	featureMap := graph.NewGraphTensor([]float32{1, 2, 3, 4}, []int{1, 2, 2, 1}, "feature_map")
	rois := graph.NewGraphTensor([]float32{0, 0, 0, 1, 1}, []int{1, 5}, "rois")
	pooled := featureMap.MaxRoiPool(rois, 2, 2, "roi_pooled")
	graph.SetOutput(pooled)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Pooled Output Shape: %v\n", pooled.value.GetShape())
	fmt.Printf("Pooled Output Data: %v\n", pooled.value.Data)

	pooled.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("FeatureMap Gradients: %v\n", featureMap.Grad().Data)
	fmt.Printf("ROIs Gradients: %v\n", rois.Grad().Data)
}

func TestMaxRoiPool_Complex(t *testing.T) {
	graph := NewComputationalGraph()

	featureMap := graph.NewGraphTensor([]float32{
		1, 11, 2, 12, 3, 13,
		4, 14, 5, 15, 6, 16,
		7, 17, 8, 18, 9, 19,
		10, 20, 11, 21, 12, 22,
		13, 23, 14, 24, 15, 25,
		16, 26, 17, 27, 18, 28,
	}, []int{2, 3, 3, 2}, "feature_map")

	rois := graph.NewGraphTensor([]float32{
		0, 0.0, 0.0, 1.0, 1.0,
		1, 0.0, 0.0, 0.5, 0.5,
	}, []int{2, 5}, "rois")

	pooled := featureMap.MaxRoiPool(rois, 2, 2, "roi_pooled")
	graph.SetOutput(pooled)

	graph.Forward()

	fmt.Println("\nComplex Test - After Forward Pass:")
	fmt.Printf("Pooled Output Shape: %v\n", pooled.value.GetShape())
	fmt.Printf("Pooled Output Data: %v\n", pooled.value.Data)

	expected := []float32{
		9, 19, 9, 19, 9, 19, 9, 19,
		14, 24, 14, 24, 14, 24, 14, 24,
	}

	for i, val := range pooled.value.Data {
		if val != expected[i] {
			t.Errorf("Output mismatch at index %d: expected %.1f, got %.1f", i, expected[i], val)
		}
	}

	pooled.Grad().Fill(1.0)
	graph.Backward()

	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("FeatureMap Gradients: %v\n", featureMap.Grad().Data)

	expectedGrad := []float32{
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 4, 4, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 4, 0, 4, 0, 0,
		0, 0, 0, 0, 0, 0,
	}

	for i, val := range featureMap.Grad().Data {
		if val != expectedGrad[i] {
			t.Errorf("Gradient mismatch at index %d: expected %.1f, got %.1f", i, expectedGrad[i], val)
		}
	}
}
