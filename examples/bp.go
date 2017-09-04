package main

import (
	"fmt"

	"math"

	"github.com/oneleo/godl/ann"
)

func main() {
	s := ann.Sigmoid{}
	fmt.Println(s.Apply(0))
	x := []int{1, 2, 3}
	fmt.Println(len(x))

	var tests = struct {
		a    [][]float64
		b    [][]float64
		want [][]float64
	}{
		[][]float64{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
		[][]float64{{7.0, 8.0}, {9.0, 10, 0}, {11.0, 12.0}},
		[][]float64{{58.0, 64.0}, {139.0, 154.0}},
	}

	//{
	//{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
	//{{7.0, 8.0}, {9.0, 10, 0}, {11.0, 12.0}},
	//{{58.0, 64.0}, {139.0, 154.0}},
	//	{{0}, {0}}, {{0}, {0}}}, {{0}}, {0}}},
	//}
	fmt.Println(tests.a[0][1])
	m := ann.Matrix{}
	out, _ := m.Dot(m.Transpose([][]float64{{1}, {1}, {1}}), [][]float64{{1}, {2}, {3}})
	fmt.Println(out)

	//b := ann.BP{}
	/*
		fmt.Println(ann.DefaultBP.Node([][]float64{{-999, -999, -999}}, [][]float64{{1, 2, 3}}))
		fmt.Println(ann.DefaultBP.Node([][]float64{{-999, -999, -999}}, [][]float64{{1, 2, 3}}))
		fmt.Println(ann.DefaultBP.Node([][]float64{{999, 999, 999}}, [][]float64{{1, 2, 3}}))
	*/
	//fmt.Println(b.Node([][]float64{{-999}, {-999}, {-999}}, [][]float64{{1}, {2}, {3}}))

	fmt.Println(math.Sqrt(4))
}
