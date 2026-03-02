using System;
using UnityEngine;

public struct FrameData
{
    public Texture currentTexture;
    public Pose currentPose;
    public Ray currentRay;
    public Vector2Int currentResolution;
    public DateTime currentTimestamp;
}