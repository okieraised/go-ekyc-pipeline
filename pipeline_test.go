package go_ekyc_pipeline

import (
	"fmt"
	"github.com/okieraised/go-ekyc-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"github.com/stretchr/testify/assert"
	"gocv.io/x/gocv"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"gorgonia.org/tensor"
	"io"
	"os"
	"testing"
)

const (
	tritonTestURL = "127.0.0.1:8301"
)

func genTestNearData() (*gocv.Mat, error) {
	f, err := os.Open("./test_data/near")
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
	f, err := os.Open("./test_data/mid")
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
	f, err := os.Open("./test_data/multiple.jpg")
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
	f, err := os.Open("./test_data/far")
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
	f, err := os.Open("./test_data/id_card")
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

func TestNewEKYCPipeline(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	pipeline, err := NewEKYCPipeline(tritonClient)
	assert.NoError(t, err)
	assert.NotNil(t, pipeline)
}

func TestEKYCPipeline_LivenessActiveCheck(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	far, err := genTestFarData()
	assert.NoError(t, err)

	mid, err := genTestMidData()
	assert.NoError(t, err)

	near, err := genTestNearData()
	assert.NoError(t, err)

	lmkNear := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			169.7128, 213.38426,
			455.29285, 223.66956,
			310.71146, 320.74503,
			195.21452, 379.8982,
			408.377, 384.25134,
		}),
	)

	lmkMid := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			276.0993, 226.09839,
			450.5989, 228.72801,
			365.71985, 283.22446,
			300.92358, 324.42694,
			427.99792, 326.67972,
		}),
	)

	lmkFar := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			266.14566, 220.05692,
			397.149, 221.45383,
			332.13202, 258.6127,
			284.05356, 294.186,
			380.22375, 294.31165,
		}),
	)

	pipeline, err := NewEKYCPipeline(tritonClient)
	assert.NoError(t, err)
	assert.NotNil(t, pipeline)

	livenessScoreCrop, livenessScoreFull, isLiveness, err := pipeline.livenessActiveCheck(*far, *mid, *near, lmkFar, lmkMid, lmkNear)
	fmt.Println("livenessScoreCrop", livenessScoreCrop)
	fmt.Println("livenessScoreFull", livenessScoreFull)
	fmt.Println("isLiveness", isLiveness)
	assert.NoError(t, err)
}

func TestEKYCPipeline_GetFaceQuality(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	far, err := genTestFarData()
	assert.NoError(t, err)

	mid, err := genTestMidData()
	assert.NoError(t, err)

	near, err := genTestNearData()
	assert.NoError(t, err)

	lmkNear := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			169.7128, 213.38426,
			455.29285, 223.66956,
			310.71146, 320.74503,
			195.21452, 379.8982,
			408.377, 384.25134,
		}),
	)

	lmkMid := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			276.0993, 226.09839,
			450.5989, 228.72801,
			365.71985, 283.22446,
			300.92358, 324.42694,
			427.99792, 326.67972,
		}),
	)

	lmkFar := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			266.14566, 220.05692,
			397.149, 221.45383,
			332.13202, 258.6127,
			284.05356, 294.186,
			380.22375, 294.31165,
		}),
	)

	pipeline, err := NewEKYCPipeline(tritonClient)
	assert.NoError(t, err)
	assert.NotNil(t, pipeline)

	score, isFaceMask, err := pipeline.getFaceQuality(*far, *mid, *near, lmkFar, lmkMid, lmkNear)
	assert.NoError(t, err)
	fmt.Println(score, isFaceMask)
}

func TestEKYCPipeline_SamePersonCheck(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	far, err := genTestFarData()
	assert.NoError(t, err)

	mid, err := genTestMidData()
	assert.NoError(t, err)

	near, err := genTestNearData()
	assert.NoError(t, err)

	lmkNear := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			169.7128, 213.38426,
			455.29285, 223.66956,
			310.71146, 320.74503,
			195.21452, 379.8982,
			408.377, 384.25134,
		}),
	)

	lmkMid := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			276.0993, 226.09839,
			450.5989, 228.72801,
			365.71985, 283.22446,
			300.92358, 324.42694,
			427.99792, 326.67972,
		}),
	)

	lmkFar := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			266.14566, 220.05692,
			397.149, 221.45383,
			332.13202, 258.6127,
			284.05356, 294.186,
			380.22375, 294.31165,
		}),
	)

	pipeline, err := NewEKYCPipeline(tritonClient)
	assert.NoError(t, err)
	assert.NotNil(t, pipeline)

	scoreFM, scoreMN, isSamePerson, err := pipeline.samePersonCheck(*far, *mid, *near, lmkFar, lmkMid, lmkNear)
	assert.NoError(t, err)
	fmt.Println(scoreFM, scoreMN, isSamePerson)
}

func TestEKYCPipeline_FaceAntiSpoofingActiveVerify(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	far, err := genTestFarData()
	assert.NoError(t, err)

	mid, err := genTestMidData()
	assert.NoError(t, err)

	near, err := genTestNearData()
	assert.NoError(t, err)

	lmkNear := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			169.7128, 213.38426,
			455.29285, 223.66956,
			310.71146, 320.74503,
			195.21452, 379.8982,
			408.377, 384.25134,
		}),
	)

	lmkMid := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			276.0993, 226.09839,
			450.5989, 228.72801,
			365.71985, 283.22446,
			300.92358, 324.42694,
			427.99792, 326.67972,
		}),
	)

	lmkFar := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			266.14566, 220.05692,
			397.149, 221.45383,
			332.13202, 258.6127,
			284.05356, 294.186,
			380.22375, 294.31165,
		}),
	)

	pipeline, err := NewEKYCPipeline(tritonClient)
	assert.NoError(t, err)
	assert.NotNil(t, pipeline)

	res, err := pipeline.FaceAntiSpoofingActiveVerify(*far, *mid, *near, lmkFar, lmkMid, lmkNear)
	assert.NoError(t, err)
	fmt.Println("res", res)
}

func TestEKYCPipeline_PersonIDCardVerify(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	far, err := genTestFarData()
	assert.NoError(t, err)

	idCard, err := genTestIDCardData()
	assert.NoError(t, err)

	lmkFar := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			266.14566, 220.05692,
			397.149, 221.45383,
			332.13202, 258.6127,
			284.05356, 294.186,
			380.22375, 294.31165,
		}),
	)

	pipeline, err := NewEKYCPipeline(tritonClient)
	assert.NoError(t, err)
	assert.NotNil(t, pipeline)

	score, isSamePerson, err := pipeline.PersonIDCardVerify(*idCard, *far, lmkFar)
	assert.NoError(t, err)
	fmt.Println(score, isSamePerson)
}

func TestEKYCPipeline_CropSelfie(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)
	assert.NoError(t, err)

	far, err := genTestFarData()
	assert.NoError(t, err)

	pipeline, err := NewEKYCPipeline(tritonClient)
	assert.NoError(t, err)
	assert.NotNil(t, pipeline)
	_, err = pipeline.CropSelfie(*far)
	assert.NoError(t, err)
}

func TestEKYCPipeline_CropFaceIDCard(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)
	assert.NoError(t, err)

	idCard, err := genTestIDCardData()
	assert.NoError(t, err)

	pipeline, err := NewEKYCPipeline(tritonClient)
	assert.NoError(t, err)
	assert.NotNil(t, pipeline)
	_, err = pipeline.CropFaceIDCard(*idCard)
	assert.NoError(t, err)
}

func TestEKYCPipeline_CropSelfie_Multiple(t *testing.T) {
	tritonClient, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	far, err := genTestMultipleFaceData()
	assert.NoError(t, err)

	pipeline, err := NewEKYCPipeline(tritonClient)
	assert.NoError(t, err)
	assert.NotNil(t, pipeline)
	_, err = pipeline.CropSelfie(*far)
	assert.NoError(t, err)
}
