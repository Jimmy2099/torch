package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"testing"
)

func TestGraphExportImport(t *testing.T) {
	originalGraph := NewComputationalGraph()

	x := originalGraph.NewTensor([]float32{2.0, 2.0}, []int{2}, "x")
	w := originalGraph.NewTensor([]float32{3.0, 3.0}, []int{2}, "w")

	wx := x.Multiply(w, "w*x")
	square := wx.Multiply(wx, "square")

	originalGraph.SetOutput(square)

	originalGraph.Forward()
	fmt.Println("Original Output:", square.Value().Data)

	const filename = "graph_export.json"
	//if err := originalGraph.ExportToFile(filename); err != nil {
	//	t.Fatalf("Export failed: %v", err)
	//}
	fmt.Println("\nGraph exported to", filename)

	importedGraph, err := LoadFromFile(filename)
	if err != nil {
		t.Fatalf("Import failed: %v", err)
	}
	fmt.Println("Graph imported from", filename)

	fmt.Println("\nImported Graph Structure:")
	//importedGraph.PrintGraph()

	importedOutput := importedGraph.GetOutput()
	if importedOutput == nil {
		t.Fatal("Imported graph has no output")
	}

	importedGraph.Forward()
	fmt.Println("\nImported Output:", importedOutput.Value().Data)

	xTensor := importedGraph.GetTensors()["x"]
	xTensor.SetValue(tensor.NewTensor([]float32{3.0, 3.0}, []int{2}))

	importedGraph.Forward()
	fmt.Println("\nImported Output with new input:", importedOutput.Value().Data)

	importedGraph.Backward()
	wTensor := importedGraph.GetTensors()["w"]
	fmt.Println("\nImported w gradient:", wTensor.Grad().Data)

	wData := wTensor.Value().Data
	wGrad := wTensor.Grad().Data
	for i := range wData {
		wData[i] -= 0.1 * wGrad[i]
	}

	importedGraph.Forward()
	fmt.Println("\nAfter update on imported graph:", importedOutput.Value().Data)
}

func TestGraphSaveRestoreState(t *testing.T) {
	graph := NewComputationalGraph()
	x := graph.NewTensor([]float32{2.0}, []int{1}, "x")
	w := graph.NewTensor([]float32{3.0}, []int{1}, "w")
	b := graph.NewTensor([]float32{1.0}, []int{1}, "b")

	wx := x.Multiply(w, "w*x")
	add := wx.Add(b, "wx+b")

	graph.SetOutput(add)

	graph.Forward()
	fmt.Println("Initial Output:", add.Value().Data)

	const stateFile = "graph_state.json"
	//if err := graph.ExportToFile(stateFile); err != nil {
	//	t.Fatalf("Failed to save state: %v", err)
	//}
	fmt.Println("Graph state saved to", stateFile)

	w.SetValue(tensor.NewTensor([]float32{4.0}, []int{1}))
	b.SetValue(tensor.NewTensor([]float32{2.0}, []int{1}))

	graph.Forward()
	fmt.Println("After modification:", add.Value().Data)

	restoredGraph, err := LoadFromFile(stateFile)
	if err != nil {
		t.Fatalf("Failed to restore state: %v", err)
	}
	fmt.Println("Graph state restored from", stateFile)

	restoredW := restoredGraph.GetTensors()["w"]
	restoredB := restoredGraph.GetTensors()["b"]
	restoredOutput := restoredGraph.GetOutput()

	fmt.Println("Restored w value:", restoredW.Value().Data)
	fmt.Println("Restored b value:", restoredB.Value().Data)

	restoredGraph.Forward()
	fmt.Println("Restored output:", restoredOutput.Value().Data)
}
