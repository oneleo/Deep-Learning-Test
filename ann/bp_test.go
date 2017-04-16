package ann

import (
	"testing"
)

func TestSigmoid(t *testing.T) {
	var tests = []struct {
		input float64
		want  float64
	}{
		{0.0, 0.5},
		{65535.0, 1.0},
		{-65536.0, 0.0},
	}
	s := Sigmoid{}
	for _, test := range tests {
		if got := s.Apply(test.input); float32(got) != float32(test.want) {
			t.Errorf("Sigmoid(%g) = %g, want %g", test.input, got, test.want)
		}
	}
}

func TestMatrix(t *testing.T) {
	var tests = []struct {
		a    [][]float64
		b    [][]float64
		want [][]float64
	}{
		{[][]float64{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
			[][]float64{{7.0, 8.0}, {9.0, 10, 0}, {11.0, 12.0}},
			[][]float64{{58.0, 64.0}, {139.0, 154.0}}},
	}
	m := Matrix{}

forBreak:
	for _, test := range tests {
		got, _ := m.Dot(test.a, test.b)
		for i := 0; i < len(got); i++ {
			for j := 0; j < len(got[0]); j++ {
				if float32(got[i][j]) != float32(test.want[i][j]) {
					t.Error("Matrix A:\n", test.a, "\n\nMatrix B:\n", test.b, "\n\nGot:\n", got, "\n\nWant:\n", test.want)
					break forBreak
				}
			}
		}
	}
}

func TestTranspose(t *testing.T) {
	var tests = []struct {
		a    [][]float64
		want [][]float64
	}{
		{[][]float64{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}},
			[][]float64{{1.0, 3.0, 5.0}, {2.0, 4.0, 6.0}}},
		{[][]float64{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}},
			[][]float64{{1.0, 3.0, 5.0, 7.0}, {2.0, 4.0, 6.0, 8.0}}},
		{[][]float64{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}},
			[][]float64{{1.0}, {2, 0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}, {9.0}}},
		{[][]float64{{1.0}, {2, 0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}, {9.0}},
			[][]float64{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}}},
	}
	m := Matrix{}
forBreak:
	for _, test := range tests {
		got := m.Transpose(test.a)
		for i := 0; i < len(got); i++ {
			for j := 0; j < len(got[0]); j++ {
				if float32(got[i][j]) != float32(test.want[i][j]) {
					t.Error("Matrix A:\n", test.a, "\n\nGot:\n", got, "\n\nWant:\n", test.want)
					break forBreak
				}
			}
		}
	}
}

func TestNode(t *testing.T) {
	var tests = []struct {
		a    [][]float64
		b    [][]float64
		want float64
	}{
		{[][]float64{{0.0, 0.0, 0.0}},
			[][]float64{{7.0, 8.0, 9.0}},
			0.5},
		{[][]float64{{1.0, 1.0, 1.0}},
			[][]float64{{999.0, 999.0, 999.0}},
			1.0},
		{[][]float64{{-1.0, -1.0, -1.0}},
			[][]float64{{999.0, 999.0, 999.0}},
			0.0},
	}
	// Select a Activation Function: BP.Act
	// X does not implement Y (... method has a pointer receiver):
	// http://stackoverflow.com/questions/40823315/go-x-does-not-implement-y-method-has-a-pointer-receiver
	b := BP{Act: &Sigmoid{}}
	for _, test := range tests {
		got := b.Node(test.a, test.b)

		if float32(got) != float32(test.want) {
			t.Error("Matrix A:\n", test.a, "\n\nMatrix B:\n", test.b, "\n\nGot:\n", got, "\n\nWant:\n", test.want)
		}
	}
}
