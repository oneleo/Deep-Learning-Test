package ann

import (
	"errors"
	"log"
	"math"
)

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

type Matrix struct {
}

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
			for k := 0; k < xColumn; k++ { // xColumn
				out[i][j] += x[i][k] * y[k][j]
			}
		}
	}
	return out, nil
}

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

type BP struct {
	In    []float64
	Train []float64
	Act   Activater
	Rate  float64
	w     [][][][]float64
	y     [][]float64
}

func (b *BP) Node(x [][]float64, w [][]float64) float64 {
	m := Matrix{}
	dotProduct, err := m.Dot(m.Transpose(x), w)
	if err != nil {
		//panic("Length of dimnames illegal to matrix dot")
		log.Fatal("Length of dimnames illegal to matrix dot\n")
	}
	if len(x) > 1 && len(w) > 1 {
		log.Fatal("Length of dimnames illegal to node output\n")
	}
	y := b.Act.Apply(dotProduct[0][0])
	return y
}
