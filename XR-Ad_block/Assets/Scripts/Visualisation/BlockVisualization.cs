/*
    Summary:
    Animates and updates the spawned blocker quad, including ID label and
    optional user-selected image material.

    Pipeline:
    BlockPlacementManager -> BlockVisualization -> blocker prefab visuals
*/
using TMPro;
using UnityEngine;

public class BlockVisualization : MonoBehaviour
{
    [Header("UI Elements")]
    [SerializeField]
    private TextMeshPro idText;

    [Header("Camera facing")]
    [SerializeField]
    private Transform labelTransform;

    [Header("Materials")]
    [SerializeField]
    private Material defaultMaterial;

    [SerializeField]
    private Material imageMaterial;

    [Header("Pop-in Animation")]
    [SerializeField]
    private float popInDuration = 0.15f;

    [Header("Smoothing")]
    [SerializeField]
    private float positionSmoothSpeed = 4f;
    private float scaleSmoothSpeed = 3f;

    [Header("Image Override")]
    [SerializeField]
    private Renderer quadRenderer;

    private Transform _cameraTransform;
    private Vector3 _targetScale;
    private float _popInTimer = 0f;
    private bool _isPopping = true;
    private Vector3 _targetPosition;
    private bool _initialized = false;

    private void Awake()
    {
        var cameraRig = FindFirstObjectByType<OVRCameraRig>();
        _cameraTransform = cameraRig != null ? cameraRig.centerEyeAnchor : Camera.main?.transform;

        _targetScale = transform.localScale;
        transform.localScale = Vector3.zero;
    }

    private void LateUpdate()
    {
        if (_isPopping)
        {
            _popInTimer += Time.deltaTime;
            float t = Mathf.Clamp01(_popInTimer / popInDuration);
            transform.localScale = _targetScale * Mathf.SmoothStep(0f, 1f, t);
            if (t >= 1f)
            {
                _isPopping = false;
            }
        }
        else
        {
            transform.localScale = Vector3.Lerp(
                transform.localScale,
                _targetScale,
                Time.deltaTime * scaleSmoothSpeed
            );
        }

        if (_initialized)
        {
            transform.position = Vector3.Lerp(
                transform.position,
                _targetPosition,
                Time.deltaTime * positionSmoothSpeed
            );
        }
        if (labelTransform != null && _cameraTransform != null)
        {
            float topEdge = 0.5f;
            labelTransform.localPosition = new Vector3(0f, topEdge + 0.08f, -0.02f);

            labelTransform.LookAt(_cameraTransform);
            labelTransform.Rotate(0f, 180f, 0f);
        }
    }

    public void SetBlockData(int id)
    {
        if (idText != null)
        {
            idText.text = $"ID: {id}";
        }
        ApplyImageOverride();
    }

    public void UpdateTargetScale(Vector3 newScale)
    {
        _targetScale = newScale;
        if (!_isPopping)
        {
            transform.localScale = _targetScale;
        }
    }

    public void SetTargetTransform(Vector3 position, Quaternion rotation, Vector3 scale)
    {
        Debug.Log(
            $"[BlockVis] SetTargetTransform called, scale: {scale}, initialized: {_initialized}"
        );
        if (!_initialized)
        {
            transform.position = position;
            transform.rotation = rotation;
            transform.localScale = Vector3.zero;
            _initialized = true;
        }
        _targetPosition = position;
        _targetScale = scale;
    }

    public void ApplyImageOverride()
    {
        if (quadRenderer == null)
        {
            Debug.LogWarning("[BlockVis] quadRenderer is null");
            return;
        }

        if (BlockerImageSettings.SelectedSprite != null && imageMaterial != null)
        {
            Material mat = new Material(imageMaterial);
            mat.SetTexture("_BaseMap", BlockerImageSettings.SelectedSprite.texture);
            quadRenderer.material = mat;
            Debug.Log(
                $"[BlockVis] Texture null? {mat.mainTexture == null}, mat shader: {mat.shader.name}"
            );
        }
        else if (defaultMaterial != null)
        {
            quadRenderer.material = defaultMaterial;
            Debug.Log($"[BlockVis] No image override, using default material");
        }
    }
}
