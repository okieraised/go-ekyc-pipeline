package config

import "time"

type FaceDetectionParams struct {
	ModelName string        `json:"model_name"`
	Mean      float64       `json:"mean"`
	Scale     float64       `json:"scale"`
	Timeout   time.Duration `json:"timeout"`
}

func NewFaceDetectionParams(modelName string, mean, scale float64, timeout time.Duration) *FaceDetectionParams {
	return &FaceDetectionParams{
		ModelName: modelName,
		Mean:      mean,
		Scale:     scale,
		Timeout:   timeout,
	}
}

var DefaultFaceDetectionParams = &FaceDetectionParams{
	ModelName: "scrfd",
	Mean:      127.5,
	Scale:     0.00784313725490196,
	Timeout:   10 * time.Second,
}

type FaceIDParams struct {
	ModelName           string        `json:"model_name"`
	Mean                float64       `json:"mean"`
	Scale               float64       `json:"scale"`
	ThresholdSameEKYC   float32       `json:"threshold_same_ekyc"`
	ThresholdSamePerson float32       `json:"threshold_same_person"`
	ImgSize             int           `json:"img_size"`
	Timeout             time.Duration `json:"timeout"`
}

func NewFaceIDParams(modelName string, mean, scale float64, thresholdSameEKYC, thresholdSamePerson float32, imgSize int, timeout time.Duration) *FaceIDParams {
	return &FaceIDParams{
		ModelName:           modelName,
		Mean:                mean,
		Scale:               scale,
		ThresholdSameEKYC:   thresholdSameEKYC,
		ThresholdSamePerson: thresholdSamePerson,
		ImgSize:             imgSize,
		Timeout:             timeout,
	}
}

var DefaultFaceIDParams = &FaceIDParams{
	ModelName:           "face_id",
	Mean:                127.5,
	Scale:               0.00784313725490196,
	ThresholdSamePerson: 0.4,
	ThresholdSameEKYC:   0.3,
	ImgSize:             112,
	Timeout:             10 * time.Second,
}

type FaceAttributeParams struct {
	ModelName         string        `json:"model_name"`
	Mean              float64       `json:"mean"`
	Scale             float64       `json:"scale"`
	ThresholdFaceMask float32       `json:"threshold_face_mask"`
	ImgSize           int           `json:"img_size"`
	Timeout           time.Duration `json:"timeout"`
}

var DefaultFaceAttributeParams = &FaceAttributeParams{
	ModelName:         "face_attribute",
	Mean:              127.5,
	Scale:             1 / 127.5,
	ThresholdFaceMask: 0.5,
	ImgSize:           128,
	Timeout:           10 * time.Second,
}

func NewFaceAttributeParams(modelName string, mean, scale float64, thresholdFaceMask float32, imgSize int, timeout time.Duration) *FaceAttributeParams {
	return &FaceAttributeParams{
		ModelName:         modelName,
		Mean:              mean,
		Scale:             scale,
		ThresholdFaceMask: thresholdFaceMask,
		ImgSize:           imgSize,
		Timeout:           timeout,
	}
}

type FaceQualityParams struct {
	ModelName      string        `json:"model_name"`
	Mean           [3]float64    `json:"mean"`
	Scale          [3]float64    `json:"scale"`
	ThresholdCover float64       `json:"threshold_cover"`
	ThresholdAll   float64       `json:"threshold_all"`
	ImgSize        int           `json:"img_size"`
	Timeout        time.Duration `json:"timeout"`
}

var DefaultFaceQualityParams = &FaceQualityParams{
	ModelName:      "face_quality_vp",
	Mean:           [3]float64{123.675, 116.28, 103.53},
	Scale:          [3]float64{1 / (0.229 * 255.0), 1 / (0.224 * 255.0), 1 / (0.225 * 255.0)},
	ThresholdCover: 0.5,
	ThresholdAll:   0.5,
	ImgSize:        112,
	Timeout:        10 * time.Second,
}

func NewFaceQualityParams(modelName string, mean, scale [3]float64, thresholdCover, thresholdAll float64, imgSize int, timeout time.Duration) *FaceQualityParams {
	return &FaceQualityParams{
		ModelName:      modelName,
		Mean:           mean,
		Scale:          scale,
		ThresholdCover: thresholdCover,
		ThresholdAll:   thresholdAll,
		ImgSize:        imgSize,
		Timeout:        timeout,
	}
}

type FaceAntiSpoofingParams struct {
	ModelName string        `json:"model_name"`
	Mean      [3]float64    `json:"mean"`
	STD       [3]float64    `json:"std"`
	Threshold float32       `json:"threshold"`
	ImgSize   int           `json:"img_size"`
	Timeout   time.Duration `json:"timeout"`
}

var DefaultCropFaceAntiSpoofingParams = &FaceAntiSpoofingParams{
	ModelName: "face_anti_spoofing_crop_l14",
	Mean:      [3]float64{0.485, 0.456, 0.406},
	STD:       [3]float64{0.229, 0.224, 0.225},
	Threshold: 0.58,
	ImgSize:   224,
	Timeout:   10 * time.Second,
}

var DefaultFullFaceAntiSpoofingParams = &FaceAntiSpoofingParams{
	ModelName: "face_anti_spoofing_fi_l14",
	Mean:      [3]float64{0.485, 0.456, 0.406},
	STD:       [3]float64{0.229, 0.224, 0.225},
	Threshold: 0.48,
	ImgSize:   224,
	Timeout:   10 * time.Second,
}

func NewFaceAntiSpoofingParams(modelName string, mean, std [3]float64, threshold float32, imgSize int, timeout time.Duration) *FaceAntiSpoofingParams {
	return &FaceAntiSpoofingParams{
		ModelName: modelName,
		Mean:      mean,
		STD:       std,
		Threshold: threshold,
		ImgSize:   imgSize,
		Timeout:   timeout,
	}
}
