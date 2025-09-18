package graph_visualize

import (
	"fmt"
	"os"
)

type Node struct {
	ID    string `json:"id"`
	Label string `json:"label"`
	Color string `json:"color,omitempty"`
}

type Edge struct {
	From  string `json:"from"`
	To    string `json:"to"`
	Label string `json:"label,omitempty"`
}

type GraphVisualize struct {
	Nodes []Node `json:"nodes"`
	Edges []Edge `json:"edges"`
}

func (m *GraphVisualize) ToDOT() string {
	dot := "digraph G {\n"
	dot += "  rankdir=LR;\n"
	dot += "  node [shape=box, style=filled];\n\n"

	for _, node := range m.Nodes {
		color := node.Color
		if color == "" {
			color = "lightblue"
		}
		dot += fmt.Sprintf("  %s [label=\"%s\", fillcolor=\"%s\"];\n", node.ID, node.Label, color)
	}

	dot += "\n"

	for _, edge := range m.Edges {
		dot += fmt.Sprintf("  %s -> %s [label=\"%s\"];\n", edge.From, edge.To, edge.Label)
	}

	dot += "}\n"
	return dot
}

func (m *GraphVisualize) Save() {
	dotContent := m.ToDOT()
	fmt.Println("DOT Format:")
	fmt.Println(dotContent)

	dotFile, _ := os.Create("graph.dot")
	dotFile.WriteString(dotContent)
	dotFile.Close()
}
