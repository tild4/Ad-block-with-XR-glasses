/*
    BlockVisualization

    PURPOSE:
    Manages the individual visual representation of a single ad-blocker.
    
    ARCHITECTURE:
    - Initialization: Scales up the block with a "Pop-in" animation.
    - UI Update: Sets the TextMeshPro text to show the object's unique ID.
    - Feedback: Includes a warning state (red blink) to notify when
      the object is about to be deleted.

    IMPORTANT:
    Uses URP-specific material properties ("_BaseColor") to change
    visuals at runtime without breaking performance.
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

    [Header("Pop-in Animation")]
    [SerializeField]
    private float popInDuration = 0.15f;

    [Header("Smoothing")]
    [SerializeField]
    private float positionSmoothSpeed = 4f;
    private float scaleSmooothSpeed = 3f;

    [Header("Image Override")]
    [SerializeField] private Renderer quadRenderer;

    private Transform _cameraTransform;
    private Vector3 _targetScale;
    private float _popInTimer = 0f;
    private bool _isPopping = true;
    private Vector3 _targetPosition;
    private bool _initialized = false;

    private void Awake()
    {
        var cameraRig = FindFirstObjectByType<OVRCameraRig>();
        _cameraTransform = cameraRig != null
            ? cameraRig.centerEyeAnchor
            : Camera.main?.transform;

        _targetScale = transform.localScale;
        transform.localScale = Vector3.zero; // Start invisible for pop-in
    }

    private void LateUpdate()
    {
        if (_isPopping)
        {
            _popInTimer += Time.deltaTime;
            float t = Mathf.Clamp01(_popInTimer / popInDuration);
            transform.localScale = _targetScale * Mathf.SmoothStep(0f, 1f, t);
            //transform.localScale = Vector3.Lerp(Vector3.zero, _targetScale, progress); // Maybe Lerp?
            if (t >= 1f)
            {
                _isPopping = false; // Animation complete
            }
        }
        else
        {
            // Smooth scale after pop-in
            transform.localScale = Vector3.Lerp(
                transform.localScale, _targetScale,
                Time.deltaTime * scaleSmooothSpeed);
        }

        if (_initialized)
        {
            // Smooth position update
            transform.position = Vector3.Lerp(
                transform.position, _targetPosition,
                Time.deltaTime * positionSmoothSpeed);
        }
        if (labelTransform != null && _cameraTransform != null)
        {
            // Position above top edge of box in local space
            float topEdge = 0.5f; // quad goes from -0.5 to +0.5 in local Y
            labelTransform.localPosition = new Vector3(0f, topEdge + 0.08f, -0.02f);

            labelTransform.LookAt(_cameraTransform);
            labelTransform.Rotate(0f, 180f, 0f); // Flip to face the camera
        }
    }

    /*
        Called to initialize the block's visual data, specifically setting the ID text.
    */
    public void SetBlockData(int id)
    {
        if (idText != null)
        {
            idText.text = $"ID: {id}";
            ApplyImageOverride();
        }
    }

    public void UpdateTargetScale(Vector3 newScale)
    {
        _targetScale = newScale;
        if (!_isPopping)
        {
            transform.localScale = _targetScale; // Instantly update if not popping
        }
    }

    public void SetTargetTransform(Vector3 position, Quaternion rotation, Vector3 scale)
    {
        if (!_initialized)
        {
            transform.position = position;
            transform.rotation = rotation;
            transform.localScale = Vector3.zero; //Let pop-in handle first frame
            _initialized = true;
        }
        _targetPosition = position;
        _targetScale = scale;
    }

    public void ApplyImageOverride()
    {
        if (quadRenderer == null) return;

        if (BlockerImageSettings.SelectedSprite != null)
        {
            // Convert sprite to texture and apply
            quadRenderer.material.mainTexture = BlockerImageSettings.SelectedSprite.texture;

            //Switch to opaque-ish so the image is visible
            Color c = quadRenderer.material.color;
            quadRenderer.material.color = new Color(c.r, c.g, c.b, 1f);
        }
        else
        {
            quadRenderer.material.mainTexture = null;
            // Restore transparent blue
            quadRenderer.material.color = new Color(0.1f, 0.4f, 0.9f, 0.6f);
        }
    }
}
