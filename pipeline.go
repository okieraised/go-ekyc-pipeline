package go_ekyc_pipeline

import (
	"errors"
	"fmt"
	"github.com/okieraised/go-ekyc-pipeline/config"
	"github.com/okieraised/go-ekyc-pipeline/modules"
	"github.com/okieraised/go-ekyc-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
	"math"
)

// EKYCPipeline defines the structure of the EKYC pipeline
type EKYCPipeline struct {
	FaceID      *modules.FaceIDClient
	FaceQuality *modules.FaceQualityClient
	FaceHelper  *modules.FaceHelperClient
	FaceASFull  *modules.FaceAntiSpoofingClient
	FaceASCrop  *modules.FaceAntiSpoofingClient
}

// NewEKYCPipeline initializes new pipelines.
func NewEKYCPipeline(tritonClient *gotritonclient.TritonGRPCClient) (*EKYCPipeline, error) {

	pipeline := &EKYCPipeline{}

	// Init face id client
	faceIDClient, err := modules.NewFaceIDClient(
		tritonClient,
		config.DefaultFaceIDParams,
	)
	if err != nil {
		return pipeline, err
	}
	pipeline.FaceID = faceIDClient

	// Init face quality client
	faceQualityClient, err := modules.NewFaceQualityClient(
		tritonClient,
		config.DefaultFaceQualityParams,
	)
	if err != nil {
		return pipeline, err
	}
	pipeline.FaceQuality = faceQualityClient

	// Init face anti-spoofing client
	faceASFullClient, err := modules.NewFaceAntiSpoofingClient(
		tritonClient,
		config.DefaultFullFaceAntiSpoofingParams,
	)
	if err != nil {
		return pipeline, err
	}
	pipeline.FaceASFull = faceASFullClient

	faceASCropClient, err := modules.NewFaceAntiSpoofingClient(
		tritonClient,
		config.DefaultCropFaceAntiSpoofingParams,
	)
	if err != nil {
		return pipeline, err
	}
	pipeline.FaceASCrop = faceASCropClient

	// Init face helper function
	faceHelper, err := modules.NewFaceHelperClient(
		tritonClient,
		faceIDClient.ModelParams.ImgSize,
		faceASCropClient.ModelParams.ImgSize,
		128,
		nil,
		nil,
		nil,
	)
	pipeline.FaceHelper = faceHelper

	return pipeline, nil
}

// similarityScore computes cosine similarity
func (c *EKYCPipeline) similarityScore(A, B *tensor.Dense) (float32, error) {
	a := A.Float32s()
	b := B.Float32s()

	if len(a) != len(b) {
		return 0, fmt.Errorf("vectors must have the same length")
	}

	var dotProduct, normA, normB float64

	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}

	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)

	if normA == 0 || normB == 0 {
		return 0, fmt.Errorf("zero vector encountered")
	}

	cosineSimilarity := dotProduct / (normA * normB)

	return float32(cosineSimilarity), nil
}

/*
embeddingExtraction extracts face features from input images and returns slices of 512 float32 elements.
Inputs:

  - images ([]gocv.Mat): input face images.

Outputs:

  - images ([]*tensor.Dense): N-Dimension Array of face embeddings.
*/
func (c *EKYCPipeline) embeddingExtraction(images []gocv.Mat) ([]*tensor.Dense, error) {
	embeddings, err := c.FaceID.InferBatch([][]gocv.Mat{images})
	if err != nil {
		return nil, err
	}

	return embeddings, nil
}

/*
getFaceQuality checks for face obstructions from 3 facial images and 3 corresponding facial landmarks.

Inputs:

  - imgFar (gocv.Mat): Capture far-distance face image.
  - imgMid (gocv.Mat): Capture mid-distance face image.
  - imgNear (gocv.Mat): Capture near-distance face image.
  - lmkFar (*tensor.Dense): imgFar landmarks.
  - lmkMid (*tensor.Dense): imgMid landmarks.
  - lmkNear (*tensor.Dense): imgNear landmarks.

Outputs:

  - maskScore (float32): Face mask score from model.
  - isFaceMask (float32): Face mask decision.
*/
func (c *EKYCPipeline) getFaceQuality(imgFar, imgMid, imgNear gocv.Mat, lmkFar, lmkMid, lmkNear *tensor.Dense) (float32, bool, error) {

	var err error
	var maskScore float32
	var isFaceMask bool

	croppedFace, _, err := c.FaceHelper.AlignWarpFaces(
		[]gocv.Mat{imgFar, imgMid, imgNear},
		[]*tensor.Dense{lmkFar, lmkMid, lmkNear},
		nil,
	)
	if err != nil {
		return maskScore, isFaceMask, err
	}

	scoreCover, err := c.FaceQuality.InferBatch([][]gocv.Mat{croppedFace})
	if err != nil {
		return maskScore, isFaceMask, err
	}
	for _, t := range scoreCover {
		data := t.Float32s()
		for _, v := range data {
			if v > maskScore {
				maskScore = v
			}
		}
	}

	return maskScore, isFaceMask, nil
}

/*
livenessActiveCheck checks face liveness from 3 facial images and 3 corresponding facial landmarks.

Inputs:

  - imgFar (gocv.Mat): Capture far-distance face image.
  - imgMid (gocv.Mat): Capture mid-distance face image.
  - imgNear (gocv.Mat): Capture near-distance face image.
  - lmkFar (*tensor.Dense): imgFar landmarks.
  - lmkMid (*tensor.Dense): imgMid landmarks.
  - lmkNear (*tensor.Dense): imgNear landmarks.

Outputs:

  - livenessScoreCrop (float32): Liveness score using crop model.
  - livenessScoreFull (float32): Liveness score using full image model.
  - isLiveness (bool): Liveness decision.
*/
func (c *EKYCPipeline) livenessActiveCheck(imgFar, imgMid, imgNear gocv.Mat, lmkFar, lmkMid, lmkNear *tensor.Dense) (float32, float32, bool, error) {

	var err error
	var livenessScoreCrop, livenessScoreFull float32
	var isLiveness bool

	croppedFaces, _, err := c.FaceHelper.AlignFASFaces(
		[]gocv.Mat{imgFar, imgMid, imgNear},
		[]*tensor.Dense{lmkFar, lmkMid, lmkNear},
		nil,
	)
	if err != nil {
		return livenessScoreCrop, livenessScoreFull, isLiveness, err
	}

	// infer fas crop
	livenessScoreCrop, err = c.FaceASCrop.InferSingle(croppedFaces[0], croppedFaces[1], croppedFaces[2])
	if err != nil {
		return livenessScoreCrop, livenessScoreFull, isLiveness, err
	}
	// infer fas full
	livenessScoreFull, err = c.FaceASFull.InferSingle(imgFar, imgMid, imgNear)
	if err != nil {
		return livenessScoreCrop, livenessScoreFull, isLiveness, err
	}
	isLiveness = (livenessScoreCrop > c.FaceASCrop.ModelParams.Threshold) && (livenessScoreFull > c.FaceASFull.ModelParams.Threshold)

	return livenessScoreCrop, livenessScoreFull, isLiveness, nil
}

/*
livenessPassiveCheck checks face liveness from 6 facial images.

Inputs:

  - fImgFar (gocv.Mat): Capture far-distance full face image.
  - fImgMid (gocv.Mat): Capture mid-distance full face image.
  - fImgNear (gocv.Mat): Capture near-distance full face image.
  - cImgFar (gocv.Mat): Capture far-distance cropped face image.
  - cImgMid (gocv.Mat): Capture mid-distance cropped face image.
  - cImgNear (gocv.Mat): Capture near-distance cropped face image.

Outputs:

  - livenessScoreCrop (float32): Liveness score using crop model.
  - livenessScoreFull (float32): Liveness score using full image model.
  - isLiveness (bool): Liveness decision.
*/
func (c *EKYCPipeline) livenessPassiveCheck(fImgFar, fImgMid, fImgNear, cImgFar, cImgMid, cImgNear gocv.Mat) (float32, float32, bool, error) {
	var err error
	var livenessScoreCrop, livenessScoreFull float32
	var isLiveness bool

	// infer fas crop
	livenessScoreCrop, err = c.FaceASCrop.InferSingle(cImgFar, cImgMid, cImgNear)
	if err != nil {
		return livenessScoreCrop, livenessScoreFull, isLiveness, err
	}
	// infer fas full
	livenessScoreFull, err = c.FaceASFull.InferSingle(fImgFar, fImgMid, fImgNear)
	if err != nil {
		return livenessScoreCrop, livenessScoreFull, isLiveness, err
	}
	isLiveness = (livenessScoreCrop > c.FaceASCrop.ModelParams.Threshold) && (livenessScoreFull > c.FaceASFull.ModelParams.Threshold)

	return livenessScoreCrop, livenessScoreFull, isLiveness, nil
}

/*
samePersonCheck verifies if the input images belong to the same person.

Inputs:

  - imgFar (gocv.Mat): Capture far-distance face image.
  - imgMid (gocv.Mat): Capture mid-distance face image.
  - imgNear (gocv.Mat): Capture near-distance face image.
  - lmkFar (*tensor.Dense): imgFar landmarks.
  - lmkMid (*tensor.Dense): imgMid landmarks.
  - lmkNear (*tensor.Dense): imgNear landmarks.

Outputs:

  - scoreFM (float32): Similarity score between far- and mid- images.
  - scoreMN (float32): Similarity score between mid- and near- images.
  - isSamePerson (bool): Similarity decision.
*/
func (c *EKYCPipeline) samePersonCheck(imgFar, imgMid, imgNear gocv.Mat, lmkFar, lmkMid, lmkNear *tensor.Dense) (float32, float32, bool, error) {

	var err error
	var scoreFM, scoreMN float32
	var isSamePerson bool

	croppedFaces, _, err := c.FaceHelper.AlignWarpFaces(
		[]gocv.Mat{imgFar, imgMid, imgNear},
		[]*tensor.Dense{lmkFar, lmkMid, lmkNear},
		nil,
	)
	if err != nil {
		return scoreFM, scoreMN, isSamePerson, err
	}

	faceEmbeddings, err := c.embeddingExtraction(croppedFaces)
	if err != nil {
		return scoreFM, scoreMN, isSamePerson, err
	}

	vFar, vMid, vNear := faceEmbeddings[0], faceEmbeddings[1], faceEmbeddings[2]

	scoreFM, err = c.similarityScore(vFar, vMid)
	if err != nil {
		return scoreFM, scoreMN, isSamePerson, err
	}
	scoreMN, err = c.similarityScore(vMid, vNear)
	if err != nil {
		return scoreFM, scoreMN, isSamePerson, err
	}

	isSamePerson = scoreFM >= c.FaceID.ModelParams.ThresholdSamePerson && scoreMN >= c.FaceID.ModelParams.ThresholdSamePerson

	return scoreFM, scoreMN, isSamePerson, nil
}

/*
GetFaceLandmarks5 returns the facial alndmarks of the input images.
Inputs:

  - images ([]gocv.Mat): input face images.

Outputs:

  - batchLandmarks: ([]*tensor.Dense): Facial landmarks from the input images.
*/
func (c *EKYCPipeline) GetFaceLandmarks5(batchImages []gocv.Mat) ([]*tensor.Dense, error) {
	_, batchLandmarks, err := c.FaceHelper.GetFaceLandmarks5(
		batchImages,
		nil,
		utils.RefPointer(true),
		nil,
		nil,
		utils.RefPointer(true),
	)
	return batchLandmarks, err
}

/*
FaceAntiSpoofingActiveVerify verifies face from input face images and landmarks.

Inputs:

  - imgFar (gocv.Mat): Capture far-distance face image.
  - imgMid (gocv.Mat): Capture mid-distance face image.
  - imgNear (gocv.Mat): Capture near-distance face image.
  - lmkFar (*tensor.Dense): imgFar landmarks.
  - lmkMid (*tensor.Dense): imgMid landmarks.
  - lmkNear (*tensor.Dense): imgNear landmarks.

Outputs:

  - scoreFM (*FaceAntiSpoofingVerify): face anti-spoofing result.
*/
func (c *EKYCPipeline) FaceAntiSpoofingActiveVerify(imgFar, imgMid, imgNear gocv.Mat, lmkFar, lmkMid, lmkNear *tensor.Dense) (*config.FaceAntiSpoofingVerify, error) {

	resp := &config.FaceAntiSpoofingVerify{
		IsFaceMask:        false,
		IsLiveness:        false,
		IsSamePerson:      false,
		ScoreMN:           -1,
		ScoreFM:           -1,
		LivenessScoreFull: -1,
		LivenessScoreCrop: -1,
		FaceMaskScore:     -1,
	}

	if lmkFar == nil {
		_, fLmks, err := c.FaceHelper.GetFaceLandmarks5([]gocv.Mat{imgFar}, nil, utils.RefPointer(true), nil, nil, utils.RefPointer(true))
		if err != nil {
			return resp, err
		}
		if len(fLmks) == 0 {
			return resp, errors.New("cannot detect any face in mid-face image")
		}
		lmkFar = fLmks[0]
	}
	if lmkMid == nil {
		_, mLmks, err := c.FaceHelper.GetFaceLandmarks5([]gocv.Mat{imgMid}, nil, utils.RefPointer(true), nil, nil, utils.RefPointer(true))
		if err != nil {
			return resp, err
		}
		if len(mLmks) == 0 {
			return resp, errors.New("cannot detect any face in mid-face image")
		}
		lmkMid = mLmks[0]
	}
	if lmkNear == nil {
		_, nLmks, err := c.FaceHelper.GetFaceLandmarks5([]gocv.Mat{imgNear}, nil, utils.RefPointer(true), nil, nil, utils.RefPointer(true))
		if err != nil {
			return resp, err
		}
		if len(nLmks) == 0 {
			return resp, errors.New("cannot detect any face in near-face image")
		}
		lmkNear = nLmks[0]
	}

	// Check same person
	scoreFM, scoreMN, isSamePerson, err := c.samePersonCheck(imgFar, imgMid, imgNear, lmkFar, lmkMid, lmkNear)
	if err != nil {
		return resp, err
	}

	resp.ScoreFM = scoreFM
	resp.ScoreMN = scoreMN
	resp.IsSamePerson = isSamePerson

	// Check face obstruction
	maskScore, isFaceMask, err := c.getFaceQuality(imgFar, imgMid, imgNear, lmkFar, lmkMid, lmkNear)
	if err != nil {
		return resp, err
	}
	resp.FaceMaskScore = maskScore
	resp.IsFaceMask = isFaceMask

	// Check liveness
	livenessScoreCrop, livenessScoreFull, isLiveness, err := c.livenessActiveCheck(imgFar, imgMid, imgNear, lmkFar, lmkMid, lmkNear)
	if err != nil {
		return resp, err
	}
	resp.LivenessScoreCrop = livenessScoreCrop
	resp.LivenessScoreFull = livenessScoreFull
	resp.IsLiveness = isLiveness

	return resp, nil
}

/*
FaceAntiSpoofingPassiveVerify verifies face from input face images.

Inputs:

  - fImgFar (gocv.Mat): Capture far-distance face image.
  - fImgMid (gocv.Mat): Capture mid-distance face image.
  - fImgNear (gocv.Mat): Capture near-distance face image.

Outputs:

  - scoreFM (*FaceAntiSpoofingVerify): face anti-spoofing result.
*/
func (c *EKYCPipeline) FaceAntiSpoofingPassiveVerify(fImgFar, fImgMid, fImgNear gocv.Mat) (*config.FaceAntiSpoofingVerify, error) {

	resp := &config.FaceAntiSpoofingVerify{
		IsFaceMask:        false,
		IsLiveness:        false,
		IsSamePerson:      false,
		ScoreMN:           -1,
		ScoreFM:           -1,
		LivenessScoreFull: -1,
		LivenessScoreCrop: -1,
		FaceMaskScore:     -1,
	}

	_, fLmks, err := c.FaceHelper.GetFaceLandmarks5([]gocv.Mat{fImgFar}, utils.RefPointer(true), nil, nil, nil, utils.RefPointer(true))
	if err != nil {
		return resp, err
	}
	if len(fLmks) == 0 {
		return resp, errors.New("cannot detect any face in far-face image")
	}
	lmkFar := fLmks[0]

	_, mLmks, err := c.FaceHelper.GetFaceLandmarks5([]gocv.Mat{fImgMid}, utils.RefPointer(true), nil, nil, nil, utils.RefPointer(true))
	if err != nil {
		return resp, err
	}
	if len(mLmks) == 0 {
		return resp, errors.New("cannot detect any face in mid-face image")
	}
	lmkMid := mLmks[0]

	_, nLmks, err := c.FaceHelper.GetFaceLandmarks5([]gocv.Mat{fImgNear}, utils.RefPointer(true), nil, nil, nil, utils.RefPointer(true))
	if err != nil {
		return resp, err
	}
	if len(nLmks) == 0 {
		return resp, errors.New("cannot detect any face in near-face image")
	}
	lmkNear := nLmks[0]

	// Check same person
	scoreFM, scoreMN, isSamePerson, err := c.samePersonCheck(fImgFar, fImgMid, fImgNear, lmkFar, lmkMid, lmkNear)
	resp.ScoreFM = scoreFM
	resp.ScoreMN = scoreMN
	resp.IsSamePerson = isSamePerson

	// Check face obstruction
	maskScore, isFaceMask, err := c.getFaceQuality(fImgFar, fImgMid, fImgNear, lmkFar, lmkMid, lmkNear)
	if err != nil {
		return resp, err
	}
	resp.FaceMaskScore = maskScore
	resp.IsFaceMask = isFaceMask

	// Check liveness
	livenessCrop, livenessFull, isLiveness, err := c.livenessActiveCheck(fImgFar, fImgMid, fImgNear, lmkFar, lmkMid, lmkNear)
	if err != nil {
		return resp, err
	}
	resp.LivenessScoreCrop = livenessCrop
	resp.LivenessScoreFull = livenessFull
	resp.IsLiveness = isLiveness

	return resp, nil
}

/*
PersonIDCardVerify checks if the face on the id card matches with input face image

Inputs:

  - cardImg (gocv.Mat): ID card with face.
  - imgFar (gocv.Mat): Capture far-distance face image.
  - lmkFar (*tensor.Dense): imgFar landmarks.

Outputs:

  - scoreFM (*FaceAntiSpoofingVerify): face anti-spoofing result.
*/
func (c *EKYCPipeline) PersonIDCardVerify(cardImg, ImgFar gocv.Mat, lmkFar *tensor.Dense) (float32, bool, error) {

	var err error
	var similarityScore float32
	var isSamePerson bool

	_, cardLmks, err := c.FaceHelper.GetFaceLandmarks5([]gocv.Mat{cardImg}, utils.RefPointer(true), nil, nil, nil, utils.RefPointer(true))
	if err != nil {
		return similarityScore, isSamePerson, err
	}

	if len(cardLmks) == 0 {
		return similarityScore, isSamePerson, errors.New("cannot detect face in card image")
	}
	cardLmk := cardLmks[0]

	if lmkFar == nil {
		_, landmarks, err := c.FaceHelper.GetFaceLandmarks5([]gocv.Mat{ImgFar}, nil, utils.RefPointer(true), nil, nil, utils.RefPointer(true))
		if err != nil {
			return similarityScore, isSamePerson, err
		}
		if len(landmarks) == 0 {
			return similarityScore, isSamePerson, errors.New("cannot detect face in input face image")
		}
		lmkFar = landmarks[0]
	}

	croppedFaces, _, err := c.FaceHelper.AlignWarpFaces([]gocv.Mat{cardImg, ImgFar}, []*tensor.Dense{cardLmk, lmkFar}, nil)

	extractions, err := c.embeddingExtraction(croppedFaces)
	if err != nil {
		return similarityScore, isSamePerson, err
	}

	vCar, vFar := extractions[0], extractions[1]
	similarityScore, err = c.similarityScore(vCar, vFar)
	if err != nil {
		return similarityScore, isSamePerson, err
	}
	isSamePerson = similarityScore >= c.FaceID.ModelParams.ThresholdSameEKYC

	return similarityScore, isSamePerson, nil
}

/*
FaceQualityVerify checks for face obstructions

Inputs:

  - imgFar (gocv.Mat): Capture far-distance face image.
  - imgMid (gocv.Mat): Capture mid-distance face image.
  - imgNear (gocv.Mat): Capture near-distance face image.
  - lmkFar (*tensor.Dense): imgFar landmarks.
  - lmkMid (*tensor.Dense): imgMid landmarks.
  - lmkNear (*tensor.Dense): imgNear landmarks.

Outputs:

  - maskScore (float32): Face mask score from model.
  - isFaceMask (float32): Face mask decision.
*/
func (c *EKYCPipeline) FaceQualityVerify(imgFar, imgMid, imgNear gocv.Mat, lmkFar, lmkMid, lmkNear *tensor.Dense) (float32, bool, error) {
	return c.getFaceQuality(imgFar, imgMid, imgNear, lmkFar, lmkMid, lmkNear)
}

/*
ExtractFaceVector extracts face from inputs image as slice of float32.

Inputs:

  - img (gocv.Mat): Capture face image.
  - lmk (*tensor.Dense): img landmarks.

Outputs:

  - vector ([]float32): Vector representations of input face image.
*/
func (c *EKYCPipeline) ExtractFaceVector(img gocv.Mat, lmk *tensor.Dense) ([]float32, error) {
	if lmk == nil {
		_, landmarks, err := c.FaceHelper.GetFaceLandmarks5([]gocv.Mat{img}, nil, utils.RefPointer(true), nil, nil, nil)
		if err != nil {
			return nil, err
		}
		if len(landmarks) == 0 {
			return nil, errors.New("cannot detect face in input face image")
		}
		lmk = landmarks[0]
	}

	croppedFaces, _, err := c.FaceHelper.AlignWarpFaces([]gocv.Mat{img}, []*tensor.Dense{lmk}, nil)
	if err != nil {
		return nil, err
	}

	t, err := c.embeddingExtraction(croppedFaces)
	if err != nil {
		return nil, err
	}

	return t[0].Float32s(), nil
}

/*
CropSelfie extracts face roi from input image.

Inputs:

  - img (gocv.Mat): Capture face image.

Outputs:

  - img (*gocv.Mat): ROI of the face.
*/
func (c *EKYCPipeline) CropSelfie(img gocv.Mat) (*gocv.Mat, error) {
	bBoxes, landmarks, err := c.FaceHelper.GetFaceLandmarks5([]gocv.Mat{img, img, img}, utils.RefPointer(true), nil, nil, nil, nil)
	if err != nil {
		return nil, err
	}

	if len(bBoxes) == 0 {
		return nil, errors.New("cannot detect face in input face image")
	}
	bbox := bBoxes[0]
	lmk := landmarks[0]
	err = lmk.Reshape(5, 2)
	if err != nil {
		return nil, err
	}
	faceImage, _, err := c.FaceHelper.AlignFaceIDCard(img, lmk, bbox, nil)
	if err != nil {
		return nil, err
	}

	rgbImage := gocv.NewMat()
	gocv.CvtColor(faceImage, &rgbImage, gocv.ColorBGRToRGB)

	return &rgbImage, nil
}

/*
CropFaceIDCard extracts face roi from input id card image.

Inputs:

  - img (gocv.Mat): Capture face image.

Outputs:

  - img (*gocv.Mat): ROI of the face.
*/
func (c *EKYCPipeline) CropFaceIDCard(img gocv.Mat) (*gocv.Mat, error) {
	imgShapes := img.Size()
	h, w := imgShapes[0], imgShapes[1]

	bBoxes, landmarks, err := c.FaceHelper.GetFaceLandmarks5([]gocv.Mat{img}, utils.RefPointer(true), nil, nil, nil, nil)
	if err != nil {
		return nil, err
	}
	if len(bBoxes) == 0 {
		return nil, errors.New("cannot detect face in input face image")
	}

	box, centerIdx, err := modules.GetCenterFace(bBoxes, nil, nil, &modules.DimensionPair[int]{
		Width:  w % 5,
		Height: h % 2,
	})

	lmk := landmarks[centerIdx]
	err = lmk.Reshape(5, 2)
	if err != nil {
		return nil, err
	}

	faceImage, _, err := c.FaceHelper.AlignFaceIDCard(img, lmk, box, nil)
	if err != nil {
		return nil, err
	}

	rgbImage := gocv.NewMat()
	gocv.CvtColor(faceImage, &rgbImage, gocv.ColorBGRToRGB)

	return &rgbImage, nil
}
