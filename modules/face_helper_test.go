package modules

import (
	"github.com/okieraised/go-ekyc-pipeline/config"
	"github.com/okieraised/go-ekyc-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"github.com/stretchr/testify/assert"
	"gocv.io/x/gocv"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	_ "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"io"
	"os"
	"testing"
)

const (
	tritonTestURL = "127.0.0.1:8301"
)

func genTestNearData() (*gocv.Mat, error) {
	f, err := os.Open("../test_data/near")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	res, err := utils.ConvertImageToMat(content)
	if err != nil {
		return nil, err
	}
	return res, nil
}

func genTestMidData() (*gocv.Mat, error) {
	f, err := os.Open("../test_data/mid")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	res, err := utils.ConvertImageToMat(content)
	if err != nil {
		return nil, err
	}
	return res, nil
}

func genTestMultipleFaceData() (*gocv.Mat, error) {
	f, err := os.Open("../test_data/multiple.jpg")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	res, err := utils.ConvertImageToMat(content)
	if err != nil {
		return nil, err
	}
	return res, nil
}

func genTestFarData() (*gocv.Mat, error) {
	f, err := os.Open("../test_data/far")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	res, err := utils.ConvertImageToMat(content)
	if err != nil {
		return nil, err
	}
	return res, nil
}

func genTestIDCardData() (*gocv.Mat, error) {
	f, err := os.Open("../test_data/id_card")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	res, err := utils.ConvertImageToMat(content)
	if err != nil {
		return nil, err
	}
	return res, nil
}

func TestNewFaceHelper(t *testing.T) {

	triton, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	mid, err := genTestMidData()
	assert.NoError(t, err)

	near, err := genTestNearData()
	assert.NoError(t, err)

	far, err := genTestFarData()
	assert.NoError(t, err)

	faceHelper, err := NewFaceHelperClient(triton, 112, 224, 128, nil, nil, nil)
	assert.NoError(t, err)
	batchBBoxes, batchLandmarks, err := faceHelper.GetFaceLandmarks5(
		[]gocv.Mat{*far, *mid, *near},
		nil,
		utils.RefPointer(true),
		utils.RefPointer(float32(0.5)),
		nil,
		utils.RefPointer(true),
	)
	assert.NoError(t, err)
	assert.NotNil(t, batchBBoxes)
	assert.NotNil(t, batchLandmarks)
}

func TestFaceHelper_AlignWarpFaces(t *testing.T) {
	triton, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	near, err := genTestNearData()
	assert.NoError(t, err)

	mid, err := genTestMidData()
	assert.NoError(t, err)

	far, err := genTestFarData()
	assert.NoError(t, err)

	faceHelper, err := NewFaceHelperClient(triton, 112, 224, 128, nil, nil, nil)
	assert.NoError(t, err)

	lmkFar := config.ConvertMetadataToTensors(&config.FaceLandmark{
		LeftEye: config.Coordinate2D{
			X: 268.21160889,
			Y: 214.80873108,
		},
		RightEye: config.Coordinate2D{
			X: 393.86187744,
			Y: 216.18772888,
		},
		Nose: config.Coordinate2D{
			X: 335.14328003,
			Y: 257.70404053,
		},
		LeftMouth: config.Coordinate2D{
			X: 284.20394897,
			Y: 294.3260498,
		},
		RightMouth: config.Coordinate2D{
			X: 385.45410156,
			Y: 294.4005127,
		},
	})

	lmkMid := config.ConvertMetadataToTensors(&config.FaceLandmark{
		LeftEye: config.Coordinate2D{
			X: 249.41821289,
			Y: 213.39225769,
		},
		RightEye: config.Coordinate2D{
			X: 483.30783081,
			Y: 219.90057373,
		},
		Nose: config.Coordinate2D{
			X: 360.6725769,
			Y: 299.56945801,
		},
		LeftMouth: config.Coordinate2D{
			X: 260.47143555,
			Y: 369.52905273,
		},
		RightMouth: config.Coordinate2D{
			X: 467.76800537,
			Y: 375.612854,
		},
	})

	lmkNear := config.ConvertMetadataToTensors(&config.FaceLandmark{
		LeftEye: config.Coordinate2D{
			X: 173.57267761,
			Y: 191.85157776,
		},
		RightEye: config.Coordinate2D{
			X: 450.2043457,
			Y: 210.12382507,
		},
		Nose: config.Coordinate2D{
			X: 309.74865723,
			Y: 302.90393066,
		},
		LeftMouth: config.Coordinate2D{
			X: 180.64160156,
			Y: 377.55731201,
		},
		RightMouth: config.Coordinate2D{
			X: 418.83895874,
			Y: 392.7986145,
		},
	})

	_, _, err = faceHelper.AlignWarpFaces([]gocv.Mat{*far, *mid, *near}, []*tensor.Dense{lmkFar, lmkMid, lmkNear}, nil)
	assert.NoError(t, err)
}
