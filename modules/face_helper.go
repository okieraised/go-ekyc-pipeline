package modules

import (
	"errors"
	"github.com/okieraised/go-ekyc-pipeline/config"
	"github.com/okieraised/go-ekyc-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
	"image"
	"image/color"
	"math"
	"slices"
)

type DimensionPair[T int | int8 | int16 | int32 | int64 | float32 | float64] struct {
	Width  T
	Height T
}

type FaceHelperClient struct {
	faceSize     [2]int
	fasSize      [2]int
	faSize       [2]int
	faceTemplate *tensor.Dense
	fasTemplate  *tensor.Dense
	faTemplate   *tensor.Dense
	faceDet      *FaceDetectionClient
}

// NewFaceHelperClient initializes a new FaceHelperClient.
func NewFaceHelperClient(
	tritonClient *gotritonclient.TritonGRPCClient,
	faceSize,
	fasSize,
	faSize int,
	faceTemplate,
	fasTemplate,
	faTemplate *tensor.Dense,
) (*FaceHelperClient, error) {

	var err error

	faceHelper := &FaceHelperClient{}

	if faceSize == 0 {
		faceSize = 112
	}
	if fasSize == 0 {
		fasSize = 224
	}
	if faSize == 0 {
		faSize = 128
	}

	faceHelper.faceSize = [2]int{faceSize, faceSize}
	faceHelper.fasSize = [2]int{fasSize, fasSize}
	faceHelper.faSize = [2]int{faSize % 4 * 3, faSize}

	if faceTemplate == nil {
		faceTemplate = tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(5, 2),
			tensor.WithBacking([]float32{
				38.2946, 51.6963,
				73.5318, 51.5014,
				56.0252, 71.7366,
				41.5493, 92.3655,
				70.7299, 92.2041,
			}),
		)
		faceTemplate, err = faceTemplate.DivScalar(float32(faceSize/112), true)
		if err != nil {
			return nil, err
		}
	}
	faceHelper.faceTemplate = faceTemplate

	if fasTemplate == nil {
		fasTemplate = tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(5, 2),
			tensor.WithBacking([]float32{
				74.01555, 90.46853,
				135.68065, 90.12745,
				105.0441, 125.539055,
				79.71127, 161.63963,
				130.77733, 161.35718,
			}),
		)
	}
	faceHelper.fasTemplate = fasTemplate

	if faTemplate == nil {
		faTemplate = tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(5, 2),
			tensor.WithBacking([]float32{
				0.34549999237060547, 0.38670000433921814,
				0.6545000076293945, 0.38670000433921814,
				0.5, 0.5386250019073486,
				0.3970000147819519, 0.6596499681472778,
				0.6030000448226929, 0.6596499681472778,
			}),
		)
	}
	faceHelper.faTemplate = faTemplate
	faceDet, err := NewFaceDetectionClient(tritonClient, config.DefaultFaceDetectionParams)
	if err != nil {
		return nil, err
	}
	faceHelper.faceDet = faceDet

	return faceHelper, nil
}

func (c *FaceHelperClient) SwapRGBs(batchImages []gocv.Mat) []gocv.Mat {

	outputs := make([]gocv.Mat, 0, len(batchImages))

	for _, srcMat := range batchImages {
		dstMat := gocv.NewMat()
		gocv.CvtColor(srcMat, &dstMat, gocv.ColorBGRToRGB)
		outputs = append(outputs, dstMat)
	}
	return outputs
}

func (c *FaceHelperClient) SwapRGB(srcMat gocv.Mat) gocv.Mat {

	dstMat := gocv.NewMat()
	gocv.CvtColor(srcMat, &dstMat, gocv.ColorBGRToRGB)

	return dstMat
}

func getLocation(val, length float32) float32 {
	if val < 0 {
		return 0
	} else if val > length {
		return length
	} else {
		return val
	}
}

func padImage(img gocv.Mat, ratio float32) (gocv.Mat, int, int) {
	dims := img.Size()
	h, w := dims[0], dims[1]
	offX := int(ratio * float32(w))
	offY := int(ratio * float32(h))

	dstMat := gocv.NewMat()
	gocv.CopyMakeBorder(img, &dstMat, offY, offY, offX, offX, gocv.BorderConstant, color.RGBA{
		R: 0,
		G: 0,
		B: 0,
		A: 0,
	})
	return dstMat, offX, offY
}

// getLargestFace returns the largest face and its index from a list of images.
func getLargestFace(detFaces []*tensor.Dense, h, w int) (*tensor.Dense, int, error) {
	faceAreas := make([]float32, 0)
	for _, detFace := range detFaces {
		det := detFace.Float32s()
		left := getLocation(det[0], float32(w))
		right := getLocation(det[2], float32(w))
		top := getLocation(det[1], float32(h))
		bottom := getLocation(det[3], float32(h))
		faceArea := (right - left) * (bottom - top)
		faceAreas = append(faceAreas, faceArea)
	}
	maxIdx := 0
	maxVal := slices.Max(faceAreas)
	for idx, val := range faceAreas {
		if val == maxVal {
			maxIdx = idx
			break
		}
	}

	return detFaces[maxIdx], maxIdx, nil
}

// GetCenterFace returns the center face and its index from a list of images.
func GetCenterFace(detFaces []*tensor.Dense, h, w *int, center *DimensionPair[int]) (*tensor.Dense, int, error) {
	if h == nil {
		h = utils.RefPointer(0)
	}

	if w == nil {
		w = utils.RefPointer(0)
	}

	var centerCoordinate *tensor.Dense
	if center != nil {
		centerCoordinate = tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(1, 2),
			tensor.WithBacking([]float32{float32(center.Width), float32(center.Height)}),
		)
	} else {
		centerCoordinate = tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(1, 2),
			tensor.WithBacking([]float32{float32(utils.DerefPointer(w) / 2), float32(utils.DerefPointer(h) / 2)}),
		)
	}

	centerDist := make([]float32, 0)
	for _, detFace := range detFaces {
		coordinates := detFace.Float32s()
		faceCenter := tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(1, 2),
			tensor.WithBacking([]float32{(coordinates[0] + coordinates[2]) / 2, (coordinates[1] + coordinates[3]) / 2}),
		)

		dist, err := utils.ComputeLinearNorm(faceCenter, centerCoordinate)
		if err != nil {
			return nil, 0, err
		}
		centerDist = append(centerDist, dist)
	}

	centerIDx := slices.Index(centerDist, slices.Min(centerDist))
	return detFaces[centerIDx], centerIDx, nil
}

// GetFaceLandmarks5 processes the input images and return the (5,2) Matrix of facial landmarks.
func (c *FaceHelperClient) GetFaceLandmarks5(
	batchImages []gocv.Mat,
	keepLargest,
	keepCenter *bool,
	scoreThreshold,
	eyeDistanceThreshold *float32,
	tryPadding *bool,
) ([]*tensor.Dense, []*tensor.Dense, error) {

	if keepLargest == nil {
		keepLargest = utils.RefPointer(false)
	}
	if keepCenter == nil {
		keepCenter = utils.RefPointer(false)
	}
	if tryPadding == nil {
		tryPadding = utils.RefPointer(false)
	}
	if scoreThreshold == nil {
		scoreThreshold = utils.RefPointer(float32(0.5))
	}

	if c.faceDet == nil {
		return nil, nil, nil
	}

	batchResults, err := c.faceDet.InferBatch([][]gocv.Mat{batchImages})
	if err != nil {
		return nil, nil, err
	}

	batchBBoxes := make([]*tensor.Dense, 0)
	batchLandmarks := make([]*tensor.Dense, 0)

	for idx := range len(batchImages) {

		results := batchResults[idx]
		filterBBoxes := make([]*tensor.Dense, 0)
		filterLandmarks := make([]*tensor.Dense, 0)

		offX, offY := 0, 0
		if utils.DerefPointer(tryPadding) && len(results) == 0 {
			var paddedImage gocv.Mat
			paddedImage, offX, offY = padImage(batchImages[idx], 0.5)
			results, err = c.faceDet.InferSingle([]gocv.Mat{paddedImage})
		}
		for _, result := range results {
			bboxOffset := tensor.New(
				tensor.Of(tensor.Float32),
				tensor.WithShape(1, 4),
				tensor.WithBacking([]float32{
					float32(offX), float32(offY),
					float32(offX), float32(offY),
				}),
			)
			bbox, err := result.Box.Sub(bboxOffset)
			if err != nil {
				return nil, nil, err
			}
			landmarkOffset := tensor.New(
				tensor.Of(tensor.Float32),
				tensor.WithShape(1, 10),
				tensor.WithBacking([]float32{
					float32(offX), float32(offY),
					float32(offX), float32(offY),
					float32(offX), float32(offY),
					float32(offX), float32(offY),
					float32(offX), float32(offY),
				}),
			)
			landmark, err := result.Landmark.Sub(landmarkOffset)
			if err != nil {
				return nil, nil, err
			}
			flattenlandmarks := landmark.Float32s()
			lm1 := tensor.New(
				tensor.Of(tensor.Float32),
				tensor.WithShape(1, 2),
				tensor.WithBacking([]float32{flattenlandmarks[0], flattenlandmarks[1]}),
			)
			lm2 := tensor.New(
				tensor.Of(tensor.Float32),
				tensor.WithShape(1, 2),
				tensor.WithBacking([]float32{flattenlandmarks[2], flattenlandmarks[3]}),
			)

			eyeDist, err := utils.ComputeLinearNorm(lm1, lm2)
			if err != nil {
				return nil, nil, err
			}
			if eyeDistanceThreshold != nil {
				if eyeDist < utils.DerefPointer(eyeDistanceThreshold) {
					continue
				}
			}
			if result.Score.Float32s()[0] < utils.DerefPointer(scoreThreshold) {
				continue
			}
			landmark = tensor.New(
				tensor.Of(tensor.Float32),
				tensor.WithShape(5, 2),
				tensor.WithBacking(
					landmark.Float32s(),
				),
			)
			filterBBoxes = append(filterBBoxes, bbox)
			filterLandmarks = append(filterLandmarks, landmark)
		}
		if len(filterBBoxes) == 0 {
			if utils.DerefPointer(keepLargest) || utils.DerefPointer(keepCenter) {
				filterBBoxes = []*tensor.Dense{}
				filterLandmarks = []*tensor.Dense{}
				batchBBoxes = append(batchBBoxes, nil)
				batchLandmarks = append(batchLandmarks, nil)
			}
			continue
		}

		var filterBBox, filterLandmark *tensor.Dense

		if utils.DerefPointer(keepLargest) {
			var largestIdx int
			shapes := batchImages[idx].Size()
			h, w := shapes[0], shapes[1]
			filterBBox, largestIdx, err = getLargestFace(filterBBoxes, h, w)
			if err != nil {
				return nil, nil, err
			}
			filterLandmark = filterLandmarks[largestIdx]

		} else if utils.DerefPointer(keepCenter) == true {
			var centerIdx int
			shapes := batchImages[idx].Size()
			h, w := shapes[0], shapes[1]
			filterBBox, centerIdx, err = GetCenterFace(filterBBoxes, utils.RefPointer(h), utils.RefPointer(w), nil)
			if err != nil {
				return nil, nil, err
			}
			filterLandmark = filterLandmarks[centerIdx]
		}
		batchBBoxes = append(batchBBoxes, filterBBox)
		batchLandmarks = append(batchLandmarks, filterLandmark)
	}
	return batchBBoxes, batchLandmarks, nil
}

//func (c *FaceHelperClient) CropFace(imgs []gocv.Mat, bboxes []*tensor.Dense, paddings []int) error {
//	if paddings == nil {
//		paddings = []int{0, 13, 0, 0}
//	}
//	pady1, pady2, padx1, padx2 := paddings[0], paddings[1], paddings[2], paddings[4]
//	for idx := range imgs {
//		img := imgs[idx]
//		bbox := bboxes[idx]
//	}
//
//	return nil
//}

// AlignWarpFaces aligns input images using input landmarks and face template.
//
// Inputs:
//
//   - inputImgs ([]gocv.Mat): list of face images.
//   - landmarks ([]*tensor.Dense): list of face landmarks.
//
// Outputs:
//
//   - croppedFaces ([]gocv.Mat): list of cropped faces.
//   - affineMatrices ([]gocv.Mat): list of affine matrices.
func (c *FaceHelperClient) AlignWarpFaces(inputImgs []gocv.Mat, landmarks []*tensor.Dense, borderMode *gocv.BorderType) ([]gocv.Mat, []gocv.Mat, error) {
	defaultBorderMode := gocv.BorderConstant
	if borderMode == nil {
		borderMode = &defaultBorderMode
	}

	affineMatrices := make([]gocv.Mat, 0, len(inputImgs))
	croppedFaces := make([]gocv.Mat, 0, len(inputImgs))

	if len(inputImgs) != len(landmarks) {
		return croppedFaces, affineMatrices, errors.New("number of input images and landmarks must be equal")
	}

	for i := 0; i < len(inputImgs); i++ {
		inputImg := inputImgs[i]
		landmark := landmarks[i]
		landmark.Float32s()
		from, err := utils.TensorToPoint2fVector(landmark)
		if err != nil {
			return croppedFaces, affineMatrices, err
		}

		to, err := utils.TensorToPoint2fVector(c.faceTemplate)
		if err != nil {
			return croppedFaces, affineMatrices, err
		}

		inliers := gocv.NewMat()
		affineMatrix := gocv.EstimateAffinePartial2DWithParams(
			from,
			to,
			inliers,
			int(gocv.HomograpyMethodLMEDS),
			3.0,
			2000,
			0.99,
			10,
		)
		err = inliers.Close()
		if err != nil {
			return croppedFaces, affineMatrices, err
		}

		affineMatrices = append(affineMatrices, affineMatrix)

		croppedFace := gocv.NewMat()
		gocv.WarpAffineWithParams(
			inputImg,
			&croppedFace,
			affineMatrix,
			image.Point{
				X: c.faceSize[0],
				Y: c.faceSize[1],
			},
			gocv.InterpolationLinear,
			*borderMode,
			color.RGBA{
				R: 0,
				G: 0,
				B: 0,
				A: 0,
			},
		)
		croppedFaces = append(croppedFaces, croppedFace)
	}
	return croppedFaces, affineMatrices, nil
}

// AlignWarpFace aligns input image using input landmark and face template.
//
// Inputs:
//
//   - inputImgs (gocv.Mat): face image.
//   - landmarks (*tensor.Dense): face landmark.
//
// Outputs:
//
//   - croppedFace (gocv.Mat): cropped face.
//   - affineMatrix (gocv.Mat): affine matrix.
func (c *FaceHelperClient) AlignWarpFace(img gocv.Mat, landmark *tensor.Dense, borderMode *gocv.BorderType) (gocv.Mat, gocv.Mat, error) {
	defaultBorderMode := gocv.BorderConstant
	if borderMode == nil {
		borderMode = &defaultBorderMode
	}

	croppedFace := gocv.NewMat()
	affineMatrix := gocv.NewMat()
	var err error

	from, err := utils.TensorToPoint2fVector(landmark)
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	to, err := utils.TensorToPoint2fVector(c.faceTemplate)
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	inliers := gocv.NewMat()
	affineMatrix = gocv.EstimateAffinePartial2DWithParams(
		from,
		to,
		inliers,
		int(gocv.HomograpyMethodLMEDS),
		3.0,
		2000,
		0.99,
		10,
	)
	err = inliers.Close()
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	gocv.WarpAffineWithParams(
		img,
		&croppedFace,
		affineMatrix,
		image.Point{
			X: c.faceSize[0],
			Y: c.faceSize[1],
		},
		gocv.InterpolationLinear,
		*borderMode,
		color.RGBA{
			R: 0,
			G: 0,
			B: 0,
			A: 0,
		},
	)

	return croppedFace, affineMatrix, nil
}

// AlignFASFaces aligns input images using input landmarks and face anti-spoofing template.
//
// Inputs:
//
//   - inputImgs ([]gocv.Mat): list of face images.
//   - landmarks ([]*tensor.Dense): list of face landmarks.
//
// Outputs:
//
//   - croppedFaces ([]gocv.Mat): list of cropped faces.
//   - affineMatrices ([]gocv.Mat): list of affine matrices.
func (c *FaceHelperClient) AlignFASFaces(inputImgs []gocv.Mat, landmarks []*tensor.Dense, borderMode *gocv.BorderType) ([]gocv.Mat, []gocv.Mat, error) {
	defaultBorderMode := gocv.BorderConstant
	if borderMode == nil {
		borderMode = &defaultBorderMode
	}

	affineMatrices := make([]gocv.Mat, 0, len(inputImgs))
	croppedFaces := make([]gocv.Mat, 0, len(inputImgs))

	if len(inputImgs) != len(landmarks) {
		return croppedFaces, affineMatrices, errors.New("number of input images and landmarks must be equal")
	}

	for i := 0; i < len(inputImgs); i++ {
		inputImg := inputImgs[i]
		landmark := landmarks[i]
		landmark.Float32s()
		from, err := utils.TensorToPoint2fVector(landmark)
		if err != nil {
			return croppedFaces, affineMatrices, err
		}

		to, err := utils.TensorToPoint2fVector(c.fasTemplate)
		if err != nil {
			return croppedFaces, affineMatrices, err
		}

		inliers := gocv.NewMat()
		affineMatrix := gocv.EstimateAffinePartial2DWithParams(
			from,
			to,
			inliers,
			int(gocv.HomograpyMethodLMEDS),
			3.0,
			2000,
			0.99,
			10,
		)
		err = inliers.Close()
		if err != nil {
			return croppedFaces, affineMatrices, err
		}

		affineMatrices = append(affineMatrices, affineMatrix)

		croppedFace := gocv.NewMat()
		gocv.WarpAffineWithParams(
			inputImg,
			&croppedFace,
			affineMatrix,
			image.Point{
				X: c.fasSize[0],
				Y: c.fasSize[1],
			},
			gocv.InterpolationLinear,
			*borderMode,
			color.RGBA{
				R: 0,
				G: 0,
				B: 0,
				A: 0,
			},
		)
		croppedFaces = append(croppedFaces, croppedFace)
	}
	return croppedFaces, affineMatrices, nil
}

// AlignFASFace aligns input image using input landmark and face anti-spoofing template.
//
// Inputs:
//
//   - inputImgs (gocv.Mat): face image.
//   - landmarks (*tensor.Dense): face landmark.
//
// Outputs:
//
//   - croppedFace (gocv.Mat): cropped face.
//   - affineMatrix (gocv.Mat): affine matrix.
func (c *FaceHelperClient) AlignFASFace(img gocv.Mat, landmark *tensor.Dense, borderMode *gocv.BorderType) (gocv.Mat, gocv.Mat, error) {
	defaultBorderMode := gocv.BorderConstant
	if borderMode == nil {
		borderMode = &defaultBorderMode
	}

	croppedFace := gocv.NewMat()
	affineMatrix := gocv.NewMat()

	from, err := utils.TensorToPoint2fVector(landmark)
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	to, err := utils.TensorToPoint2fVector(c.fasTemplate)
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	inliers := gocv.NewMat()
	affineMatrix = gocv.EstimateAffinePartial2DWithParams(
		from,
		to,
		inliers,
		int(gocv.HomograpyMethodLMEDS),
		3.0,
		2000,
		0.99,
		10,
	)
	err = inliers.Close()
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	croppedFace = gocv.NewMat()
	gocv.WarpAffineWithParams(
		img,
		&croppedFace,
		affineMatrix,
		image.Point{
			X: c.fasSize[0],
			Y: c.fasSize[1],
		},
		gocv.InterpolationLinear,
		*borderMode,
		color.RGBA{
			R: 0,
			G: 0,
			B: 0,
			A: 0,
		},
	)

	return croppedFace, affineMatrix, nil

}

// AlignFaceIDCard crops and aligns the face roi in the input image.
//
// Inputs:
//
//   - img (gocv.Mat): face image.
//   - lmk (*tensor.Dense): face landmark.
//   - lmk (*tensor.Dense): face bounding box.
//
// Outputs:
//
//   - croppedFace (gocv.Mat): cropped face.
//   - affineMatrix (gocv.Mat): affine matrix.
func (c *FaceHelperClient) AlignFaceIDCard(img gocv.Mat, lmk, bbox *tensor.Dense, borderMode *gocv.BorderType) (gocv.Mat, gocv.Mat, error) {
	defaultBorderMode := gocv.BorderConstant
	if borderMode == nil {
		borderMode = &defaultBorderMode
	}

	croppedFace := gocv.NewMat()
	affineMatrix := gocv.NewMat()

	faceTemplateAdult := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			87.56117786, 140.95207892,
			152.12076214, 140.5917773,
			120.04617, 177.99955243,
			93.52425322, 216.13514054,
			146.98728108, 215.83676865,
		}),
	)

	faceTemplateBaby := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			89.26848429, 149.95460108,
			150.43019571, 149.6132627,
			120.04374, 185.05220757,
			94.91771357, 221.18065946,
			145.56689786, 220.89799136,
		}),
	)

	err := bbox.Reshape(2, 2)
	if err != nil {
		return croppedFace, affineMatrix, err
	}
	data := bbox.Float32s()

	center := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(1, 2),
		tensor.WithBacking([]float32{(data[0] + data[2]) / 2, (data[1] + data[3]) / 2}),
	)

	lmk0, err := lmk.Slice(tensor.S(0))
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	lmk1, err := lmk.Slice(tensor.S(1))
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	lmk2, err := lmk.Slice(tensor.S(2))
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	subLmk01, err := utils.Subtract(lmk0.(*tensor.Dense), lmk1.(*tensor.Dense))
	if err != nil {
		return croppedFace, affineMatrix, err
	}
	subLmk21, err := utils.Subtract(lmk2.(*tensor.Dense), lmk1.(*tensor.Dense))
	if err != nil {
		return croppedFace, affineMatrix, err
	}
	subCenterLmk1, err := utils.Subtract(center, lmk1.(*tensor.Dense))
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	// calculate dCBoxEyes
	crossProduct, err := utils.Cross2D(subLmk01, subCenterLmk1)
	if err != nil {
		return croppedFace, affineMatrix, err
	}
	norms, err := utils.ComputeLinearNorm(lmk0.(*tensor.Dense), lmk1.(*tensor.Dense))
	if err != nil {
		return croppedFace, affineMatrix, err
	}
	dCBoxEyes := crossProduct / norms

	// calculate dNoseEyes
	crossProduct, err = utils.Cross2D(subLmk01, subLmk21)
	if err != nil {
		return croppedFace, affineMatrix, err
	}
	norms, err = utils.ComputeLinearNorm(lmk0.(*tensor.Dense), lmk1.(*tensor.Dense))
	if err != nil {
		return croppedFace, affineMatrix, err
	}
	dNoseEyes := crossProduct / norms

	var ratioAdult float32 = 0.565
	var ratioBaby float32 = 0.306

	faceRatio := float32(math.Min(math.Max(float64(dCBoxEyes/dNoseEyes), float64(ratioBaby)), float64(ratioAdult)))

	scaledFaceTemplateBaby, err := faceTemplateBaby.MulScalar(ratioAdult, false)
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	scaledFaceTemplateAdult, err := faceTemplateAdult.MulScalar(ratioBaby, false)
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	subScaledTemplate, err := scaledFaceTemplateBaby.Sub(scaledFaceTemplateAdult)
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	subTemplate, err := faceTemplateAdult.Sub(faceTemplateBaby)
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	scaledSubTemplate, err := subTemplate.MulScalar(faceRatio, false)
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	numerator, err := subScaledTemplate.Add(scaledSubTemplate)
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	denominator := ratioAdult - ratioBaby

	faceTemplate, err := numerator.DivScalar(denominator, true)
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	from, err := utils.TensorToPoint2fVector(lmk)
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	to, err := utils.TensorToPoint2fVector(faceTemplate)
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	inliers := gocv.NewMat()
	affineMatrix = gocv.EstimateAffinePartial2DWithParams(
		from,
		to,
		inliers,
		int(gocv.HomograpyMethodLMEDS),
		3.0,
		2000,
		0.99,
		10,
	)
	err = inliers.Close()
	if err != nil {
		return croppedFace, affineMatrix, err
	}

	gocv.WarpAffineWithParams(
		img,
		&croppedFace,
		affineMatrix,
		image.Point{
			X: 240,
			Y: 320,
		},
		gocv.InterpolationLinear,
		*borderMode,
		color.RGBA{
			R: 0,
			G: 0,
			B: 0,
			A: 0,
		},
	)
	return croppedFace, affineMatrix, nil
}
