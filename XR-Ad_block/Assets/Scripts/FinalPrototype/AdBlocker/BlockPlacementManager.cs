/*
    Summary:
    Places, updates, and removes world-space blocker objects for tracked ads
    that are currently marked for blocking.

    Pipeline:
    TrackingManager -> BlockPlacementManager -> BlockVisualization

    Note:
    This project uses and adapts sample code provided through the Meta XR SDK.

    Copyright © Meta Platform Technologies, LLC and its affiliates.
    All rights reserved.
*/
using System.Collections.Generic;
using Meta.XR;
using UnityEngine;

public class BlockPlacementManager : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField]
    private PassthroughCameraAccess cameraAccess;

    [SerializeField]
    private EnvironmentRaycastManager realRaycastManager;

    [SerializeField]
    private OVRCameraRig cameraRig;

    [SerializeField]
    private TrackingManager trackingManager;

    [Header("Prefab")]
    [SerializeField]
    private GameObject blockPrefab;

    [Header("Settings")]
    [SerializeField]
    private bool useSpatialAnchors = true;

    [SerializeField]
    private bool useCameraPlanePlacement = true;

    // Nudges camera-plane blocks toward the viewer to reduce surface clipping.
    [SerializeField]
    private float placementPlaneOffsetMeters = 0.01f;

    [SerializeField]
    private bool logRaycastMisses;

    private readonly Dictionary<int, GameObject> activeBlocks = new Dictionary<int, GameObject>();
    private readonly Dictionary<int, OVRSpatialAnchor> activeSpatialAnchors =
        new Dictionary<int, OVRSpatialAnchor>();

    private void OnEnable()
    {
        if (trackingManager != null)
        {
            trackingManager.onTrackedObjectsUpdated += SyncBlocksWithTracking;
        }
    }

    private void OnDisable()
    {
        if (trackingManager != null)
        {
            trackingManager.onTrackedObjectsUpdated -= SyncBlocksWithTracking;
        }
    }

    private void SyncBlocksWithTracking(List<TrackedObject> activeTracks)
    {
        PipelineProfiler.begin("5. Block Placement (3D)");

        foreach (var obj in activeTracks)
        {
            PlaceOrUpdateBlock(obj);
        }

        List<int> idsToRemove = new List<int>();
        foreach (var id in activeBlocks.Keys)
        {
            bool stillTracked = activeTracks.Exists(obj => obj.id == id);
            if (!stillTracked)
            {
                idsToRemove.Add(id);
            }
        }

        foreach (var id in idsToRemove)
        {
            RemoveBlock(id);
        }

        PipelineProfiler.set("ActiveBlocks", activeBlocks.Count);
        PipelineProfiler.end("5. Block Placement (3D)");
    }

    private static Rect ToViewportRect(Rect yoloNormalizedRect)
    {
        float viewportYMin = 1f - yoloNormalizedRect.yMax;
        return new Rect(
            yoloNormalizedRect.xMin,
            viewportYMin,
            yoloNormalizedRect.width,
            yoloNormalizedRect.height
        );
    }

    private bool TryRaycastEnvironment(Ray ray, out EnvironmentRaycastHit hit)
    {
        if (realRaycastManager != null)
        {
            return realRaycastManager.Raycast(ray, out hit);
        }

        hit = default;
        return false;
    }

    private bool TryComputeCameraPlanePlacement(
        Rect yoloNormalizedRect,
        Pose cameraPose,
        Vector3 depthPoint,
        out Vector3 worldPosition,
        out Quaternion worldRotation
    )
    {
        worldPosition = default;
        worldRotation = default;

        if (cameraAccess == null || !cameraAccess.enabled || !cameraAccess.IsPlaying)
        {
            return false;
        }

        float distance = Vector3.Distance(cameraPose.position, depthPoint);
        if (distance <= 0.0001f)
        {
            return false;
        }

        Rect viewportRect = ToViewportRect(yoloNormalizedRect);
        Ray centerRay = cameraAccess.ViewportPointToRay(viewportRect.center, cameraPose);
        worldPosition = centerRay.GetPoint(distance);

        Vector3 fromCamera = (worldPosition - cameraPose.position);
        if (fromCamera.sqrMagnitude < 1e-6f)
        {
            return false;
        }

        Vector3 towardCamera = (-fromCamera).normalized;
        worldRotation = Quaternion.LookRotation(towardCamera, Vector3.up);
        worldPosition += towardCamera * placementPlaneOffsetMeters;
        return true;
    }

    private Vector2 ComputeWorldSize(Rect yoloRect, Pose cameraPose, float depth)
    {
        float halfFovRad = Mathf.Deg2Rad * (Camera.main.fieldOfView / 2f);
        float viewportWidthAtDepth = 2f * depth * Mathf.Tan(halfFovRad);
        float viewportHeightAtDepth =
            viewportWidthAtDepth
            * ((float)cameraAccess.CurrentResolution.y / cameraAccess.CurrentResolution.x);

        float padding = 0.7f;
        return new Vector2(
            yoloRect.width * viewportWidthAtDepth * padding,
            yoloRect.height * viewportHeightAtDepth * padding
        );
    }

    private void PlaceOrUpdateBlock(TrackedObject obj)
    {
        try
        {
            Debug.Log(
                $"[Block] Attempting placement for object {obj.id}, shouldBlock={obj.shouldBlock}"
            );

            if (!obj.shouldBlock && activeBlocks.ContainsKey(obj.id))
            {
                RemoveBlock(obj.id);
                return;
            }

            if (!obj.shouldBlock)
            {
                return;
            }

            if (Camera.main == null && (cameraAccess == null || !cameraAccess.enabled))
            {
                Debug.LogError("No camera available for ViewportPointToRay.");
                return;
            }

            Rect yoloRect = obj.lastDetection.bboxNormalized;
            Pose cameraPose = obj.lastDetection.frame.currentPose;
            Rect viewportRect = ToViewportRect(yoloRect);
            bool usingPassthroughRay =
                cameraAccess != null && cameraAccess.enabled && cameraAccess.IsPlaying;

            Ray ray = usingPassthroughRay
                ? cameraAccess.ViewportPointToRay(viewportRect.center, cameraPose)
                : Camera.main.ViewportPointToRay(
                    new Vector3(viewportRect.center.x, viewportRect.center.y, 0f)
                );

            if (!TryRaycastEnvironment(ray, out EnvironmentRaycastHit hit))
            {
                if (logRaycastMisses)
                {
                    Debug.LogWarning($"Raycast failed for object {obj.id}");
                }
                return;
            }

            Vector3 position;
            Quaternion rotation;
            if (
                useCameraPlanePlacement
                && TryComputeCameraPlanePlacement(
                    yoloRect,
                    cameraPose,
                    hit.point,
                    out position,
                    out rotation
                )
            ) { }
            else
            {
                position = hit.point;
                Vector3 towardCamera =
                    (usingPassthroughRay ? cameraPose.position : Camera.main.transform.position)
                    - position;
                if (towardCamera.sqrMagnitude < 1e-6f)
                {
                    towardCamera = -ray.direction;
                }
                rotation = Quaternion.LookRotation(towardCamera.normalized, Vector3.up);
            }

            float depth = Vector3.Distance(cameraPose.position, hit.point);
            Vector2 worldSize = ComputeWorldSize(yoloRect, cameraPose, depth);

            if (!activeBlocks.ContainsKey(obj.id))
            {
                CreateBlockWithAnchor(obj, position, rotation, worldSize);
            }
            else
            {
                UpdateBlock(obj, position, rotation, worldSize);
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[Block] Exception in PlaceOrUpdateBlock: {e.Message}\n{e.StackTrace}");
        }
    }

    private void CreateBlockWithAnchor(
        TrackedObject obj,
        Vector3 position,
        Quaternion rotation,
        Vector2 size
    )
    {
        GameObject block = Instantiate(blockPrefab);
        block.name = $"Block_{obj.id}";
        Vector3 worldScale = new Vector3(size.x, size.y, 0.01f);

        BlockVisualization vis = block.GetComponent<BlockVisualization>();
        if (vis != null)
        {
            vis.SetBlockData(obj.id);
            vis.SetTargetTransform(position, rotation, worldScale);
        }
        else
        {
            block.transform.position = position;
            block.transform.rotation = rotation;
            block.transform.localScale = worldScale;
        }

        if (cameraRig != null)
        {
            block.transform.SetParent(cameraRig.trackingSpace);
        }

        if (useSpatialAnchors && cameraRig != null && !activeSpatialAnchors.ContainsKey(obj.id))
        {
            OVRSpatialAnchor spatialAnchor = block.AddComponent<OVRSpatialAnchor>();
            spatialAnchor.Save(
                (anchor, success) =>
                {
                    if (!success)
                    {
                        Debug.LogWarning($"Failed to save spatial anchor for object {obj.id}");
                    }
                }
            );

            activeSpatialAnchors[obj.id] = spatialAnchor;
        }

        activeBlocks[obj.id] = block;
    }

    private void UpdateBlock(TrackedObject obj, Vector3 position, Quaternion rotation, Vector2 size)
    {
        GameObject block = activeBlocks[obj.id];
        block.transform.position = position;
        block.transform.rotation = rotation;

        Vector3 worldScale = new Vector3(size.x, size.y, 0.01f);
        BlockVisualization vis = block.GetComponent<BlockVisualization>();
        if (vis != null)
        {
            vis.SetTargetTransform(position, rotation, worldScale);
        }
        else
        {
            block.transform.position = position;
            block.transform.rotation = rotation;
            block.transform.localScale = worldScale;
        }
    }

    private void RemoveBlock(int objectId)
    {
        if (!activeBlocks.ContainsKey(objectId))
        {
            return;
        }

        if (activeSpatialAnchors.ContainsKey(objectId))
        {
            OVRSpatialAnchor anchor = activeSpatialAnchors[objectId];
            anchor.Erase((anchorToErase, success) => { });
            activeSpatialAnchors.Remove(objectId);
        }

        Destroy(activeBlocks[objectId]);
        activeBlocks.Remove(objectId);
    }

    private void OnDestroy()
    {
        var allIds = new List<int>(activeBlocks.Keys);
        foreach (int id in allIds)
        {
            RemoveBlock(id);
        }
    }

    public void ClearAllBlocks()
    {
        var allIds = new List<int>(activeBlocks.Keys);
        foreach (int id in allIds)
        {
            RemoveBlock(id);
        }
    }
}
