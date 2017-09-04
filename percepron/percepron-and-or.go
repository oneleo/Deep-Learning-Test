// Reference: https://www.zybuluo.com/hanbingtao/note/433855
// Reference: https://github.com/fisproject/go-perceptron
package main

import (
	"fmt"
)

func main() {
	// Initialise
	rate := 0.1
	bias := 0.0
	w := []float64{0.0, 0.0}
	p := Perceptron{w, bias}

	// Data input
	data := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}

	// And gate output
	label := []float64{0, 0, 0, 1}
	// Or gate output
	//label := []float64{0, 1, 1, 1}

	// 10 times for every train
	p.Train(data, label, 10, rate)

	fmt.Println("\nTrain Done!")

	// Predict unknown input
	x := []float64{0.0, 1.0}
	y := p.Predict(x)
	fmt.Println("Output: ", y)
	fmt.Println("Weights: ", p.Weights)
	fmt.Println("Bias: ", p.Bias)
}

type Perceptron struct {
	Weights []float64
	Bias    float64
}

func (self *Perceptron) Predict(input []float64) float64 {
	return f(ProductAddBias(input, self.Weights, self.Bias))
}

func (self *Perceptron) UpdateWeights(x []float64 /*DataSet*/, l float64, rate float64, actualOutput float64) {

	delta := float64(l) - actualOutput
	for i := 0; i < len(x); i++ {
		self.Weights[i] += rate * delta * x[i]
	}
	self.Bias += rate * delta
}

func (self *Perceptron) Train(d [][]float64 /*DataSet*/, l []float64, time int, rate float64) {
	for z := 0; z < time; z++ {
		fmt.Println("Weight: ", self.Weights, "\t", "Bias: ", self.Bias)
		for i := 0; i < len(d); i++ {
			t := d[i]

			//actualOutput := InnerProduct(t, self.Weights, self.Bias)
			actualOutput := self.Predict(t)
			self.UpdateWeights(t, l[i], rate, actualOutput)
			fmt.Println("data: ", t, "\t", "actualOutput: ", actualOutput)
		}
		fmt.Println()
	}
}

func f(i float64) (o float64) {
	if i > 0 {
		o = 1.0
	} else {
		o = 0.0
	}
	return o
}

// Matrix Inner product and add bias
func ProductAddBias(x []float64, w []float64, bias float64) (r float64) {
	for i := 0; i < len(x); i++ {
		r += x[i] * w[i]
	}
	r += bias
	return r
}
