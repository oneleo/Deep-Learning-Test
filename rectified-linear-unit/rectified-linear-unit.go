// Reference: https://www.zybuluo.com/hanbingtao/note/448086
package main

import (
	"fmt"
)

func main() {
	// Initialise
	rate := 0.01
	bias := 0.0
	w := []float64{0.0}
	p := Perceptron{w, bias}

	// Data input
	data := [][]float64{{5}, {3}, {8}, {1.4}, {10.1}}

	// And gate output
	//label := []float64{0, 0, 0, 1}
	// Or gate output
	label := []float64{5500, 2300, 7600, 1800, 11400}

	// 10 times for every train
	p.Train(data, label, 10, rate)

	fmt.Println("\nTrain Done!")

	// Predict unknown input
	x := []float64{6.3}
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
		for i := 0; i < len(d); i++ {
			t := d[i]

			//actualOutput := InnerProduct(t, self.Weights, self.Bias)
			actualOutput := self.Predict(t)
			self.UpdateWeights(t, l[i], rate, actualOutput)
			fmt.Println("data: ", t, "\t", "actualOutput: ", actualOutput)
		}
	}
}

func f(i float64) (o float64) {
	o = i
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
