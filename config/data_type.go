package config

import (
	"gorgonia.org/tensor"
)

type FaceDetectionOutput struct {
	Box      *tensor.Dense
	Score    *tensor.Dense
	ClassID  *tensor.Dense
	Landmark *tensor.Dense
}

type Size struct {
	Width  int
	Height int
}

func (s *Size) Max() int {
	if s.Height > s.Width {
		return s.Height
	}
	return s.Width
}

func (s *Size) Min() int {
	if s.Height < s.Width {
		return s.Height
	}
	return s.Width
}

type FaceLandmarkMetadata struct {
	Near FaceLandmark
	Mid  FaceLandmark
	Far  FaceLandmark
}

type FaceLandmark struct {
	LeftEye    Coordinate2D
	RightEye   Coordinate2D
	Nose       Coordinate2D
	LeftMouth  Coordinate2D
	RightMouth Coordinate2D
}

type Coordinate2D struct {
	X float32
	Y float32
}

func ConvertMetadataToTensors(meta *FaceLandmark) *tensor.Dense {
	tMeta := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking(
			[]float32{
				meta.LeftEye.X, meta.LeftEye.Y,
				meta.RightEye.X, meta.RightEye.Y,
				meta.Nose.X, meta.Nose.Y,
				meta.LeftMouth.X, meta.LeftMouth.Y,
				meta.RightMouth.X, meta.RightMouth.Y,
			},
		),
	)

	return tMeta
}

// FaceAntiSpoofingVerify defines the structure of the face anti-spoofing check.
type FaceAntiSpoofingVerify struct {
	IsFaceMask        bool    `json:"is_face_mask"`        // IsFaceMask determines if the face is obstructed.
	IsLiveness        bool    `json:"is_liveness"`         // IsLiveness determines if the face is real.
	IsSamePerson      bool    `json:"is_same_person"`      // IsSamePerson determines if the face images belong to the same person.
	ScoreMN           float32 `json:"score_mn"`            // ScoreMN is the similarity score between mid- and near- face image.
	ScoreFM           float32 `json:"score_fm"`            // ScoreFM is the similarity score between far- and mid- face image.
	LivenessScoreFull float32 `json:"liveness_score_full"` // LivenessScoreFull is the liveness score using full face model.
	LivenessScoreCrop float32 `json:"liveness_score_crop"` // LivenessScoreCrop is the liveness score using crop face model.
	SimilarityScore   float32 `json:"similarity_score"`    // SimilarityScore is the cosine similarity score between far-face and id card image.
	FaceMaskScore     float32 `json:"face_mask_score"`     // FaceMaskScore is the obstruction score from the model.
}
