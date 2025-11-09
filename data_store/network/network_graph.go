package network

type Node struct {
	Name    string
	Type    string
	Inputs  []*Node
	Outputs []*Node
	Parent  *Node
}

func (n *Node) ConnectInput(input *Node) {
	n.Inputs = append(n.Inputs, input)
	input.Outputs = append(input.Outputs, n)
}

func (n *Node) AddOutput(output *Node) {
	n.Outputs = append(n.Outputs, output)
	output.Parent = n
}

func (n *Node) AddInput(input *Node) {
	n.Inputs = append(n.Inputs, input)
}

type Network struct {
	nodes   []*Node
	inputs  []*Node
	outputs []*Node
}

func (g *Network) NewNode() *Node {
	n := &Node{}
	g.nodes = append(g.nodes, n)
	return n
}

func (g *Network) GetNodeByName(name string) *Node {
	for _, n := range g.nodes {
		if n.Name == name {
			return n
		}
	}
	return nil
}

func NewNetwork() *Network {
	return &Network{}
}

func (g *Network) GetInput() []*Node {
	return g.inputs
}
func (g *Network) GetOutput() []*Node {
	return g.outputs
}

func (g *Network) GetNodes() []*Node {
	return g.nodes
}

func (g *Network) AddOutput(output *Node) {
	g.outputs = append(g.outputs, output)
}

func (g *Network) AddInput(input *Node) {
	g.inputs = append(g.inputs, input)
}
