package compute_dependency_graph

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/Jimmy2099/torch/data_store/network"
)

func TestComputeDependencyGraph(t *testing.T) {
	net1 := network.NewNetwork()
	a1 := net1.NewNode()
	a1.Name = "InputA"
	b1 := net1.NewNode()
	b1.Name = "OpB"
	c1 := net1.NewNode()
	c1.Name = "OpC"
	b1.ConnectInput(a1)
	c1.ConnectInput(b1)
	net1.AddInput(a1)
	net1.AddOutput(c1)
	expectedNames1 := []string{"InputA", "OpB", "OpC"}

	net2 := network.NewNetwork()
	a2 := net2.NewNode()
	a2.Name = "A"
	b2 := net2.NewNode()
	b2.Name = "B"
	c2 := net2.NewNode()
	c2.Name = "C"
	d2 := net2.NewNode()
	d2.Name = "D"
	b2.ConnectInput(a2)
	c2.ConnectInput(a2)
	d2.ConnectInput(b2)
	d2.ConnectInput(c2)
	net2.AddInput(a2)
	net2.AddOutput(d2)
	expectedLen2 := 4

	net3 := network.NewNetwork()
	in1 := net3.NewNode()
	in1.Name = "In1"
	in2 := net3.NewNode()
	in2.Name = "In2"
	op1 := net3.NewNode()
	op1.Name = "Op1"
	op2 := net3.NewNode()
	op2.Name = "Op2"
	out1 := net3.NewNode()
	out1.Name = "Out1"
	op1.ConnectInput(in1)
	op1.ConnectInput(in2)
	op2.ConnectInput(in2)
	out1.ConnectInput(op1)
	net3.AddInput(in1)
	net3.AddInput(in2)
	net3.AddOutput(out1)
	net3.AddOutput(op2)
	expectedLen3 := 5

	net4 := network.NewNetwork()
	expectedLen4 := 0

	net5 := network.NewNetwork()
	a5 := net5.NewNode()
	a5.Name = "Single"
	net5.AddInput(a5)
	net5.AddOutput(a5)
	expectedNames5 := []string{"Single"}

	testCases := []struct {
		name          string
		net           *network.Network
		expectedLen   int
		expectedNames []string
	}{
		{
			name:          "Linear Dependency Graph",
			net:           net1,
			expectedLen:   len(expectedNames1),
			expectedNames: expectedNames1,
		},
		{
			name:        "Branch Merge Graph (DAG)",
			net:         net2,
			expectedLen: expectedLen2,
		},
		{
			name:        "Multiple Inputs Multiple Outputs",
			net:         net3,
			expectedLen: expectedLen3,
		},
		{
			name:        "Empty Network",
			net:         net4,
			expectedLen: expectedLen4,
		},
		{
			name:          "Single Node Network",
			net:           net5,
			expectedLen:   len(expectedNames5),
			expectedNames: expectedNames5,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			g := NewComputeDependencyGraph(tc.net)
			g.ComputeSortedNodes()

			var planNames []string
			for _, n := range g.GetOutputSortedNodes() {
				planNames = append(planNames, n.Name)
			}
			fmt.Printf("Test Case [%s] - Computed Order: %s\n", tc.name, strings.Join(planNames, ", "))

			if len(g.GetOutputSortedNodes()) != tc.expectedLen {
				t.Errorf("Expected %d nodes, but got %d", tc.expectedLen, len(g.GetOutputSortedNodes()))
			}

			if tc.expectedNames != nil {
				if !reflect.DeepEqual(planNames, tc.expectedNames) {
					t.Errorf("Incorrect order computed:\nExpected: %+v\nGot: %+v", tc.expectedNames, planNames)
				}
			}

			err := g.Validate()
			if err != nil {
				t.Errorf("Error in test case [%s]: %v", tc.name, err)
			}
		})
	}
}
