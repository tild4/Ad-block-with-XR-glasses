using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

/*
    ============================================================
    NEW FILE ADDED FOR CLEANUP ONLY
    ------------------------------------------------------------
    PURPOSE:
    Replaces repeated nested tuple payloads in the OCR pipeline
    with named types so the later stages are easier to read.

    NOTE:
    - This file is a cleanup addition only.
    - It does not change runtime logic.
    ============================================================
*/

public readonly struct YoloRoiTensor
{
    public readonly Tensor<float> Tensor;
    public readonly FrameData Frame;
    public readonly Rect Bounds;

    public YoloRoiTensor(Tensor<float> tensor, FrameData frame, Rect bounds)
    {
        Tensor = tensor;
        Frame = frame;
        Bounds = bounds;
    }
}

public readonly struct CroppedTextRoi
{
    public readonly Tensor<float> Tensor;
    public readonly Rect Bounds;

    public CroppedTextRoi(Tensor<float> tensor, Rect bounds)
    {
        Tensor = tensor;
        Bounds = bounds;
    }
}

public readonly struct FrameRoiBatch
{
    public readonly List<CroppedTextRoi> Rois;
    public readonly FrameData Frame;

    public FrameRoiBatch(List<CroppedTextRoi> rois, FrameData frame)
    {
        Rois = rois;
        Frame = frame;
    }
}
