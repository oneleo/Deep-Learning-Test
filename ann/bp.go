package ann

import (
	"errors"
	"log"
	"math"
	"math/rand"
	"time"
)

// Activater interface is Activation function that need have two methods.
type Activater interface {
	Apply(float64) float64
	Derivative(float64) float64
}

// Sigmoid represents sigmoid activation function
type Sigmoid struct {
}

// Apply calculates sigmoid of given value
func (s *Sigmoid) Apply(value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

// Derivative calculates sigmoid derivative of given value
func (s *Sigmoid) Derivative(value float64) float64 {
	tmp := s.Apply(value)
	return tmp * (1 - tmp)
}

// Matrix calculation set
type Matrix struct {
}

// Dot method calculate Matrix dot production.
func (m *Matrix) Dot(x [][]float64, y [][]float64) ([][]float64, error) {
	xRow, xColumn := len(x), len(x[0])
	yRow, yColumn := len(y), len(y[0])

	if xColumn != yRow {
		return nil, errors.New("Length of dimnames illegal to matrix dot")
	}
	// non-constant array bound: http://stackoverflow.com/questions/23290858/how-to-allocate-a-non-constant-sized-array-in-go
	//var out [xm][yn]float64

	// multi-dimension slice: http://stackoverflow.com/questions/39804861/what-is-a-concise-way-to-create-a-2d-slice-in-go
	out := make([][]float64, xRow)
	for i := range out {
		out[i] = make([]float64, yColumn)
	}

	for i := 0; i < xRow; i++ { // xRow
		for j := 0; j < yColumn; j++ { // yColumn
			for k := 0; k < xColumn; k++ { // xColumn or yRow are have to same.
				out[i][j] += x[i][k] * y[k][j]
			}
		}
	}
	return out, nil
}

// Transpose calculate a Matrix transpose.
func (m *Matrix) Transpose(x [][]float64) [][]float64 {
	xRow, xColumn := len(x), len(x[0])
	out := make([][]float64, xColumn)
	for i := range out {
		out[i] = make([]float64, xRow)
	}
	for i := 0; i < xColumn; i++ {
		for j := 0; j < xRow; j++ {
			out[i][j] = x[j][i]
		}
	}
	return out
}

// OneColumnRowDot calculate only one row and one column matrix
func (m *Matrix) OneColumnRowDot(x []float64, y []float64) float64 {
	if len(x) != len(y) {
		log.Fatal("Length of dimnames illegal to one row and one column Matrix dot\n")
	}
	var out float64
	for i := range x {
		out += x[i] * y[i]
	}
	return out
}

type Node struct {
	Act Activater
	m   Matrix
}

// Default Node variable, Refference: https://golang.org/pkg/go/build/#Context
var DefaultNode Node = defaultNode()

func defaultNode() Node {

	var n Node
	// Select a Activation Function: BP.Act = Sigmoid.Apply
	// X does not implement Y (... method has a pointer receiver):
	// http://stackoverflow.com/questions/40823315/go-x-does-not-implement-y-method-has-a-pointer-receiver
	n.Act = &Sigmoid{}
	return n
}

// node method calculate a single Perceptron node output.
func (n *Node) node(x [][]float64, w [][]float64) float64 {
	// Matrix A's Column have to match Matrix B's Row.
	dotProduct, err := n.m.Dot(n.m.Transpose(x), w)
	if err != nil {
		//panic("Length of dimnames illegal to matrix dot")
		log.Fatal("Length of dimnames illegal to matrix dot\n")
	}
	if len(x) > 1 && len(w) > 1 {
		log.Fatal("Length of dimnames illegal to node output\n")
	}
	y := n.Act.Apply(dotProduct[0][0])
	return y
}

// quickNode method calculate a single Perceptron node output using OneColumnRowDot method.
func (n *Node) quickNode(x []float64, w []float64) float64 {
	return n.Act.Apply(n.m.OneColumnRowDot(x, w))
}

// BP struct is the Back-propagation calculation set.
type BP struct {
	In           []float64
	Want         []float64
	Act          Activater
	Rate         float64
	Iteration    int
	HideLayerNum int
	hideNodeNum  int
	// Number of out elememts is must to equal to Want elements
	out []float64
	// Weight
	w [][]float64
	// Output of all Node
	y [][]float64
	// Include Matrix Struct
	m Matrix
	// Include Node Struct
	n Node
	// Every Nodes' δ value
	eta [][]float64
}

func (b *BP) CalcuNode() {
	rs := rand.NewSource(time.Now().UnixNano())
	r := rand.New(rs)
	if b.Act == nil {
		b.Act = &Sigmoid{}
	}
	if b.Rate == 0 {
		b.Rate = 0.3
	}
	if b.Iteration == 0 {
		b.Iteration = 1
	}
	if b.HideLayerNum == 0 {
		b.HideLayerNum = 1
	}
	if b.In == nil {
		log.Fatal("No Input elements\n")
	}
	if b.Want == nil {
		log.Fatal("No Want Output elements\n")
	}
	// Hide node number are recommand Sqrt(input number * output number).
	b.hideNodeNum = int(math.Sqrt(float64(len(b.In) * len(b.Want))))
	b.y = make([][]float64, b.HideLayerNum+2)
	// Number of Input + One 偏置項
	b.y[0] = make([]float64, len(b.In)+1)
	// Number of Output + One 偏置項
	b.y[len(b.y)-1] = make([]float64, len(b.Want)+1)
	// Number of all of hide layer + One 偏置項
	for i := 1; i < b.HideLayerNum+1; i++ {
		b.y[i] = make([]float64, b.hideNodeNum+1)
	}
	// y[?][0] = 1 as 偏置项
	for i, _ := range b.y {
		b.y[i][0] = 1
	}
	// y[0][1:] as Input
	for i, e := range b.In {
		b.y[0][i+1] = e
	}

	// Output are no weight
	b.w = make([][]float64, b.HideLayerNum+2-1)
	// number of input layer weights
	b.w[0] = make([]float64, len(b.y[0])*(len(b.y[1])-1))
	// Initial w
	for i := 0; i < len(b.w[0]); i++ {
		b.w[0][i] = r.Float64()*0.2 - 0.1
	}
	// number of output layer weights
	b.w[len(b.w)-1] = make([]float64, len(b.y[len(b.y)-2])*(len(b.y[len(b.y)-1])-1))
	// Initial w
	for i := 0; i < len(b.w[len(b.w)-1]); i++ {
		b.w[len(b.w)-1][i] = r.Float64()*0.2 - 0.1
	}
	for i := 1; i < b.HideLayerNum; i++ {
		b.w[i] = make([]float64, len(b.y[1])*(len(b.y[1])-1))
		// Initial w
		for j := 0; j < len(b.w[i]); j++ {
			b.w[i][j] = r.Float64()*0.2 - 0.1
		}
	}

	// Calculate the Output of Hide Layer one Node
	for i := 1; i < b.HideLayerNum+1; i++ {
		for j := 1; j < b.hideNodeNum+1; j++ {
			//y[i][j]=
		}
	}
	//...
}

func (b *BP) CalcuEta() {
	/*for l := 1; l < len(b.Want); l++ {
		b.eta[2][l] = (b.Want[l] - b.y[2][l]) * b.Act.Derivative(b.n.quickNode(b.y[1], b.w[2][l][1]))
	}*/
	//...
}
