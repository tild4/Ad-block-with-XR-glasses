/*
    Summary:
    Carries a sampled camera texture, pose, and resolution through the
    detection and tracking pipeline.
*/

using UnityEngine;

public struct FrameData
{
    public Texture currentTexture;
    public Pose currentPose;
    public Vector2Int currentResolution;
}
