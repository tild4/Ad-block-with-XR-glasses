using TMPro;
using UnityEngine;

public class GlobalClock : MonoBehaviour
{
    [SerializeField] private Vector3 positionOffset = new Vector3(0.25f, -0.18f, 0.5f);
    [SerializeField] private float fontSize = 3f;

    private TextMeshPro tmp;
    private Transform cameraTransform;
    private float elapsed;

    private void Start()
    {
        var cameraRig = FindFirstObjectByType<OVRCameraRig>();
        cameraTransform = cameraRig != null
            ? cameraRig.centerEyeAnchor
            : Camera.main?.transform;

        // Create text
        var textGo = new GameObject("ClockText");
        textGo.transform.SetParent(transform);
        tmp = textGo.AddComponent<TextMeshPro>();
        tmp.fontSize = fontSize;
        tmp.fontStyle = FontStyles.Bold;
        tmp.alignment = TextAlignmentOptions.Center;
        tmp.color = Color.white;
        tmp.rectTransform.sizeDelta = new Vector2(8f, 2f);

        // Background
        var bg = GameObject.CreatePrimitive(PrimitiveType.Quad);
        bg.name = "ClockBG";
        Destroy(bg.GetComponent<Collider>());
        bg.transform.SetParent(tmp.transform, false);
        bg.transform.localPosition = new Vector3(0f, 0f, 0.01f);
        bg.transform.localRotation = Quaternion.identity;
        bg.transform.localScale = new Vector3(8f, 1.5f, 1f);
        var rend = bg.GetComponent<Renderer>();
        rend.material = new Material(Shader.Find("Sprites/Default"));
        rend.material.color = new Color(0f, 0f, 0f, 0.9f);
    }

    private void LateUpdate()
    {
        if (cameraTransform == null || tmp == null) return;

        elapsed += Time.deltaTime;

        int minutes = (int)(elapsed / 60f);
        int seconds = (int)(elapsed % 60f);
        int ms = (int)((elapsed * 1000f) % 1000f);
        tmp.text = $"{minutes:D2}:{seconds:D2}.{ms:D3}";

        // Follow camera: bottom-right of view
        transform.position = cameraTransform.position
            + cameraTransform.right * positionOffset.x
            + cameraTransform.up * positionOffset.y
            + cameraTransform.forward * positionOffset.z;
        transform.rotation = cameraTransform.rotation;
    }
}
