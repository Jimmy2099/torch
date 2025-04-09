package main

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
	"math/rand"
	"time"
)

// y =a + wx+ wx + wx + b

// NeuronCellUnit represents a neuron, storing weights, bias, forward propagation output, and error term
type NeuronCellUnit struct {
	weights []float32
	bias    float32
	output  float32
	delta   float32
}

// Layer represents a layer of neurons
type Layer struct {
	neurons []NeuronCellUnit
}

// Network represents the entire neural network, composed of multiple layers
type Network struct {
	layers []Layer
}

// sigmoid is the activation function
func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidDerivative is the derivative of sigmoid, used for backpropagation
func sigmoidDerivative(output float32) float32 {
	return output * (1 - output)
}

// newNeuron randomly initializes a neuron, numInputs is the number of inputs for this neuron
func newNeuron(numInputs int) NeuronCellUnit {
	weights := make([]float32, numInputs)
	for i := range weights {
		weights[i] = rand.Float32()*0.1 - 0.05 // Small random values
	}
	bias := rand.Float32()*0.1 - 0.05
	return NeuronCellUnit{weights: weights, bias: bias}
}

// newNetwork initializes a neural network with input size, a list of hidden layer neuron counts, and output neuron count
func newNetwork(numInputs int, hiddenLayerSizes []int, numOutputs int) *Network {
	net := &Network{}
	previousSize := numInputs

	// Add hidden layers
	for _, size := range hiddenLayerSizes {
		layer := Layer{neurons: make([]NeuronCellUnit, size)}
		for i := 0; i < size; i++ {
			layer.neurons[i] = newNeuron(previousSize)
		}
		net.layers = append(net.layers, layer)
		previousSize = size
	}

	// Add output layer
	outLayer := Layer{neurons: make([]NeuronCellUnit, numOutputs)}
	for i := 0; i < numOutputs; i++ {
		outLayer.neurons[i] = newNeuron(previousSize)
	}
	net.layers = append(net.layers, outLayer)
	return net
}

// forward performs one forward propagation and returns the final output
func (net *Network) forward(input []float32) []float32 {
	outputs := input
	for li := 0; li < len(net.layers); li++ {
		layer := &net.layers[li]
		newOutputs := make([]float32, len(layer.neurons))
		for ni := 0; ni < len(layer.neurons); ni++ {
			sum := layer.neurons[ni].bias
			for wi := 0; wi < len(layer.neurons[ni].weights); wi++ {
				sum += layer.neurons[ni].weights[wi] * outputs[wi]
			}
			layer.neurons[ni].output = sigmoid(sum)
			newOutputs[ni] = layer.neurons[ni].output
		}
		outputs = newOutputs
	}
	return outputs
}

// backward calculates the errors for each layer using backpropagation and updates weights and biases
func (net *Network) backward(expected []float32, learningRate float32, inputs []float32) {
	// Calculate delta for output layer
	outputLayerIndex := len(net.layers) - 1
	outputLayer := &net.layers[outputLayerIndex]
	for i := 0; i < len(outputLayer.neurons); i++ {
		output := outputLayer.neurons[i].output
		errorVal := expected[i] - output
		outputLayer.neurons[i].delta = errorVal * sigmoidDerivative(output)
	}

	// Calculate delta for hidden layers in reverse order
	for li := outputLayerIndex - 1; li >= 0; li-- {
		layer := &net.layers[li]
		nextLayer := net.layers[li+1]
		for i := 0; i < len(layer.neurons); i++ {
			var errorSum float32
			for j := 0; j < len(nextLayer.neurons); j++ {
				errorSum += nextLayer.neurons[j].weights[i] * nextLayer.neurons[j].delta
			}
			layer.neurons[i].delta = errorSum * sigmoidDerivative(layer.neurons[i].output)
		}
	}

	// Update weights and biases for all layers
	for li := 0; li < len(net.layers); li++ {
		var layerInputs []float32
		if li == 0 {
			layerInputs = inputs
		} else {
			prevLayer := net.layers[li-1]
			layerInputs = make([]float32, len(prevLayer.neurons))
			for i := 0; i < len(prevLayer.neurons); i++ {
				layerInputs[i] = prevLayer.neurons[i].output
			}
		}
		for ni := 0; ni < len(net.layers[li].neurons); ni++ {
			for wi := 0; wi < len(net.layers[li].neurons[ni].weights); wi++ {
				net.layers[li].neurons[ni].weights[wi] += learningRate * net.layers[li].neurons[ni].delta * layerInputs[wi]
			}
			net.layers[li].neurons[ni].bias += learningRate * net.layers[li].neurons[ni].delta
		}
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Generate data: randomly generate points in a 2D plane, and classify using the equation y = 2x + 1
	// If the y-coordinate is greater than 2x+1, the label is 1; otherwise, it's 0
	numPoints := 200
	var inputs [][]float32
	var expectedOutputs [][]float32

	// Randomly generate points within the x, y range
	for i := 0; i < numPoints; i++ {
		x := rand.Float32()*10 - 5 // x in [-5, 5]
		y := rand.Float32()*10 - 5 // y in [-5, 5]
		inputs = append(inputs, []float32{x, y})
		label := float32(0.0)
		if y > 2*x+1 {
			label = 1.0
		}
		expectedOutputs = append(expectedOutputs, []float32{label})
	}

	// Create a neural network with 2 inputs, one hidden layer (2 neurons), and 1 output for binary classification
	net := newNetwork(2, []int{2}, 1)

	epochs := 10000
	learningRate := float32(0.5)

	// Training process
	for epoch := 0; epoch < epochs; epoch++ {
		// Optionally shuffle the data order
		for i := 0; i < len(inputs); i++ {
			net.forward(inputs[i])
			net.backward(expectedOutputs[i], learningRate, inputs[i])
		}
	}

	// Test the trained network: output for a subset of points
	for i := 0; i < 10; i++ {
		test := inputs[i]
		output := net.forward(test)
		prediction := 0
		if output[0] > 0.5 {
			prediction = 1
		}
		fmt.Printf("Input point: (%.2f, %.2f), Predicted classification: %d, Expected classification: %d\n", test[0], test[1], prediction, int(expectedOutputs[i][0]))
	}
}
