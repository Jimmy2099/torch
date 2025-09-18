package graph_visualize

import (
	"testing"
)

func TestSave(t *testing.T) {
	graph := GraphVisualize{
		Nodes: []Node{
			{ID: "A", Label: "Data Source", Color: "#FF6B6B"},
			{ID: "B", Label: "Processor", Color: "#4ECDC4"},
			{ID: "C", Label: "Storage", Color: "#45B7D1"},
			{ID: "D", Label: "API", Color: "#96CEB4"},
		},
		Edges: []Edge{
			{From: "A", To: "B", Label: "raw data"},
			{From: "B", To: "C", Label: "processed"},
			{From: "C", To: "D", Label: "query"},
			{From: "D", To: "A", Label: "request"},
		},
	}

	graph.Save()
}
