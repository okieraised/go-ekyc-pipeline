package utils

import (
	"fmt"
	"gorgonia.org/tensor"
)

func ComputeLinearNorm(t1, t2 *tensor.Dense) (float32, error) {
	diff, err := tensor.Sub(t1, t2)
	if err != nil {
		return 0, err
	}

	squaredDiff, err := tensor.Square(diff)
	if err != nil {

		return 0, err
	}
	sumOfSquares, err := tensor.Sum(squaredDiff)
	if err != nil {
		return 0, err
	}
	eyeDist, err := tensor.Sqrt(sumOfSquares)
	if err != nil {
		return 0, err
	}
	return eyeDist.(*tensor.Dense).Float32s()[0], nil
}

func Subtract(a, b *tensor.Dense) (*tensor.Dense, error) {
	return a.Sub(b)
}

func Cross2D(a, b *tensor.Dense) (float32, error) {
	dataA := a.Float32s()
	dataB := b.Float32s()

	if len(dataA) != 2 || len(dataB) != 2 {
		return 0, fmt.Errorf("vectors must have a size of 2")
	}
	return dataA[0]*dataB[1] - dataA[1]*dataB[0], nil
}
