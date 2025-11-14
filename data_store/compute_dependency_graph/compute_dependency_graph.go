package compute_dependency_graph

import (
	"container/list"
	"errors"
	"fmt"
	"github.com/Jimmy2099/torch/data_store/network"
)

// ComputeDependencyGraph Convert output-driven graph to input-driven graph for shape inference
type ComputeDependencyGraph struct {
	inputNetwork *network.Network
	sortedNodes  []*network.Node
}

func NewComputeDependencyGraph(net *network.Network) *ComputeDependencyGraph {
	return &ComputeDependencyGraph{inputNetwork: net}
}

func (m *ComputeDependencyGraph) traceDependencies(outputs []*network.Node) []*network.Node {
	visited := map[*network.Node]bool{}
	var required []*network.Node
	queue := list.New()

	for _, out := range outputs {
		queue.PushBack(out)
	}

	for queue.Len() > 0 {
		elem := queue.Front()
		queue.Remove(elem)
		n := elem.Value.(*network.Node)

		if visited[n] {
			continue
		}
		visited[n] = true
		required = append(required, n)

		for _, in := range n.Inputs {
			if !visited[in] {
				queue.PushBack(in)
			}
		}
	}

	return required
}

func (m *ComputeDependencyGraph) topologicalSort(nodes []*network.Node) []*network.Node {
	deps := map[*network.Node]int{}
	children := map[*network.Node][]*network.Node{}

	for _, n := range nodes {
		deps[n] = len(n.Inputs)
		for _, input := range n.Inputs {
			children[input] = append(children[input], n)
		}
	}

	queue := list.New()
	for _, n := range nodes {
		if deps[n] == 0 {
			queue.PushBack(n)
		}
	}

	var sorted []*network.Node
	for queue.Len() > 0 {
		elem := queue.Front()
		queue.Remove(elem)
		n := elem.Value.(*network.Node)

		sorted = append(sorted, n)

		for _, child := range children[n] {
			deps[child]--
			if deps[child] == 0 {
				queue.PushBack(child)
			}
		}
	}

	return sorted
}

func (m *ComputeDependencyGraph) ComputeSortedNodes() {
	required := m.traceDependencies(m.inputNetwork.GetOutput())
	sorted := m.topologicalSort(required)
	m.sortedNodes = sorted
}

func (m *ComputeDependencyGraph) GetOutputSortedNodes() []*network.Node {
	return m.sortedNodes
}

func (m *ComputeDependencyGraph) Validate() error {
	nodePositions := make(map[*network.Node]int)
	for i, node := range m.sortedNodes {
		nodePositions[node] = i
	}

	for i, node := range m.sortedNodes {
		for _, input := range node.Inputs {
			inputPos, found := nodePositions[input]
			if !found {
				return errors.New(fmt.Sprintf("Node %q's dependency %q not found in plan\n", node.Name, input.Name))
				//continue
			}
			if inputPos >= i {
				return errors.New(fmt.Sprintf("Incorrect order: Node %q (position %d) appears before its dependency %q (position %d)", node.Name, i, input.Name, inputPos))
			}
		}
	}
	return nil
}
