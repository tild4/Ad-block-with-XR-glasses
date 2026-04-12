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

    private Transform _cameraTransform;
    private Vector3 _targetScale;
    private float _popInTimer = 0f;
    private bool _isPopping = true;

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
        if (labelTransform != null && _cameraTransform != null)
        {
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
}
