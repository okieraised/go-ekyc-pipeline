package utils

import (
	"fmt"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
	"image/jpeg"
	"os"
)

func ConvertImageToMat(bImage []byte) (*gocv.Mat, error) {
	dstMat := gocv.NewMat()
	srcMat, err := gocv.IMDecode(bImage, gocv.IMReadColor)
	if err != nil {
		return &dstMat, err
	}

	gocv.CvtColor(srcMat, &dstMat, gocv.ColorBGRToRGB)
	return &dstMat, nil
}

func TensorToPoints(t *tensor.Dense) ([]gocv.Point2f, error) {
	shape := t.Shape()
	if len(shape) != 2 || shape[1] != 2 {
		return nil, fmt.Errorf("expected a 2D tensor with shape (n, 2), got shape: %v", shape)
	}
	data := t.Float32s()
	n := shape[0]
	points := make([]gocv.Point2f, n)
	for i := 0; i < n; i++ {
		points[i] = gocv.Point2f{
			X: data[i*2],
			Y: data[i*2+1],
		}
	}

	return points, nil
}

func TensorToPoint2fVector(t *tensor.Dense) (gocv.Point2fVector, error) {
	points, err := TensorToPoints(t)
	if err != nil {
		return gocv.NewPoint2fVector(), err
	}
	pointVector := gocv.NewPoint2fVectorFromPoints(points)
	return pointVector, nil
}

func AffineMatrixToTensor(mat gocv.Mat) (*tensor.Dense, error) {

	rows := mat.Rows()
	cols := mat.Cols()
	channels := mat.Channels()

	data := mat.ToBytes()

	floatData := BytesToT64[float64](data)

	var shape []int
	if channels > 1 {
		shape = []int{rows, cols, channels}
	} else {
		shape = []int{rows, cols}
	}

	t := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(shape...),
		tensor.WithBacking(floatData),
	)
	return t, nil
}

func OpenCVImageToJPEG(fPath string, jpegQuality int, img gocv.Mat) error {
	outImg, err := img.ToImage()
	if err != nil {
		return err
	}

	f, err := os.Create(fPath)
	if err != nil {
		return err
	}
	defer f.Close()

	opt := jpeg.Options{
		Quality: jpegQuality,
	}
	err = jpeg.Encode(f, outImg, &opt)
	if err != nil {
		return err
	}
	return nil
}
