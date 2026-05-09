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

    [Header("Materials")]
    [SerializeField] private Material defaultMaterial;
    [SerializeField] private Material imageMaterial;

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
    private GameObject _debugBg;

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
        }
        ApplyImageOverride();
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
        Debug.Log($"[BlockVis] SetTargetTransform called, scale: {scale}, initialized: {_initialized}");
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

    public void UpdateDecisionDebug(TrackedObject obj)
    {
        if (idText == null) return;

        // Make text larger and bold
        idText.fontSize = 6;
        idText.fontStyle = FontStyles.Bold;
        idText.rectTransform.sizeDelta = new Vector2(24f, 10f);

        if (!obj.isAnalyzed)
        {
            idText.text = $"ID:{obj.id} [YOLO] block={obj.shouldBlock}\nconf={obj.lastDetection.confidence:F2}";
            idText.color = Color.yellow;
        }
        else if (obj.nlpScores != null && obj.nlpScores.Length >= 3)
        {
            string status = obj.decisionChanged ? "CHANGED" : "CONFIRMED";
            string ocrText = string.IsNullOrEmpty(obj.text) ? "(no text)" : obj.text;
            idText.text = $"ID:{obj.id} [{status}]\n"
                + $"non-ad:{obj.nlpScores[0]:F2} benef:{obj.nlpScores[1]:F2} ad:{obj.nlpScores[2]:F2}\n"
                + $"\"{ocrText}\"";
            idText.color = obj.decisionChanged ? Color.red : Color.green;
        }
        else
        {
            string status = obj.decisionChanged ? "CHANGED" : "CONFIRMED";
            idText.text = $"ID:{obj.id} [{status}]\n{obj.decisionSource} (no text detected)";
            idText.color = obj.decisionChanged ? Color.red : Color.green;
        }

        // Create/update solid black background quad behind text
        EnsureDebugBackground();
    }

    private void EnsureDebugBackground()
    {
        if (_debugBg == null)
        {
            _debugBg = GameObject.CreatePrimitive(PrimitiveType.Quad);
            _debugBg.name = "DebugBG";
            Destroy(_debugBg.GetComponent<Collider>());
            _debugBg.transform.SetParent(idText.transform, false);
            _debugBg.transform.localPosition = new Vector3(0f, 0f, 0.05f);
            _debugBg.transform.localRotation = Quaternion.identity;

            var rend = _debugBg.GetComponent<Renderer>();
            rend.material = new Material(Shader.Find("Sprites/Default"));
            rend.material.color = new Color(0f, 0f, 0f, 0.9f);
        }

        // Size and position background to match actual text bounds
        idText.ForceMeshUpdate();
        Vector2 rendered = idText.GetRenderedValues(true);
        float pad = 1.5f;
        _debugBg.transform.localScale = new Vector3(rendered.x + pad, rendered.y + pad, 1f);

        // Center background on the rendered text area
        var bounds = idText.textBounds;
        _debugBg.transform.localPosition = new Vector3(bounds.center.x, bounds.center.y, 0.05f);
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
            Debug.Log($"[BlockVis] Texture null? {mat.mainTexture == null}, mat shader: {mat.shader.name}");

        }
        else if (defaultMaterial != null)
        {
            quadRenderer.material = defaultMaterial;
            Debug.Log($"[BlockVis] No image override, using default material");
        }

    }
}
