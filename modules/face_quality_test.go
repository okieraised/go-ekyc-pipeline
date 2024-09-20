package modules

import (
	"fmt"
	"github.com/okieraised/go-ekyc-pipeline/config"
	"github.com/okieraised/go-ekyc-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"github.com/stretchr/testify/assert"
	"gocv.io/x/gocv"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"gorgonia.org/tensor"
	"testing"
)

func TestFaceQualityClient_InferBatch(t *testing.T) {
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

	_, batchLandmarks, err := faceHelper.GetFaceLandmarks5(
		[]gocv.Mat{*far, *mid, *near},
		nil,
		utils.RefPointer(true),
		nil,
		nil,
		utils.RefPointer(true),
	)
	assert.NoError(t, err)

	croppedFaces, _, err := faceHelper.AlignWarpFaces([]gocv.Mat{*far, *mid, *near}, []*tensor.Dense{
		batchLandmarks[0],
		batchLandmarks[1],
		batchLandmarks[2],
	}, nil)
	if err != nil {
		return
	}
	assert.NoError(t, err)

	faceQualityClient, err := NewFaceQualityClient(triton, config.DefaultFaceQualityParams)
	assert.NoError(t, err)

	res, err := faceQualityClient.InferBatch([][]gocv.Mat{croppedFaces})
	assert.NoError(t, err)
	fmt.Println("res", res)

	res2, err := faceQualityClient.InferSingle(croppedFaces)
	assert.NoError(t, err)
	fmt.Println("res2", res2)
}
