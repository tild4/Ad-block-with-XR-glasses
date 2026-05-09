using TMPro;
using UnityEngine;

public class FloatingDebugText : MonoBehaviour
{
    private float lifetime = 2.5f;
    private float elapsed;
    private Transform cameraTransform;
    private TextMeshPro tmp;
    private Renderer bgRenderer;

    public static void Spawn(Vector3 position, TrackedObject obj)
    {
        var go = new GameObject($"FloatingDebug_{obj.id}");
        // Place at same direction as block but pushed out to ~10m from camera
        var cam = Camera.main;
        if (cam != null)
        {
            Vector3 dir = (position - cam.transform.position).normalized;
            go.transform.position = cam.transform.position + dir * 10f;
        }
        else
        {
            go.transform.position = position;
        }

        var tmp = go.AddComponent<TextMeshPro>();
        tmp.fontSize = 6.0f;
        tmp.fontStyle = FontStyles.Bold;
        tmp.alignment = TextAlignmentOptions.Center;
        tmp.color = Color.red;
        tmp.sortingOrder = 100;
        tmp.rectTransform.sizeDelta = new Vector2(10f, 5f);

        string ocrText = string.IsNullOrEmpty(obj.text) ? "(no text)" : obj.text;

        if (obj.nlpScores != null && obj.nlpScores.Length >= 3)
        {
            tmp.text = "CHANGED -> UNBLOCKED\n"
                + $"non-ad:{obj.nlpScores[0]:F2} benef:{obj.nlpScores[1]:F2} ad:{obj.nlpScores[2]:F2}\n"
                + $"\"{ocrText}\"";
        }
        else
        {
            tmp.text = $"CHANGED -> UNBLOCKED\n{obj.decisionSource}\n\"{ocrText}\"";
        }

        // Solid background quad
        var bg = GameObject.CreatePrimitive(PrimitiveType.Quad);
        bg.name = "BG";
        Destroy(bg.GetComponent<Collider>());
        bg.transform.SetParent(tmp.transform, false);
        bg.transform.localRotation = Quaternion.identity;
        var bgRend = bg.GetComponent<Renderer>();
        bgRend.material = new Material(Shader.Find("Sprites/Default"));
        bgRend.material.color = new Color(0f, 0f, 0f, 0.9f);

        // Size and center background on rendered text
        tmp.ForceMeshUpdate();
        Vector2 rendered = tmp.GetRenderedValues(true);
        float pad = 1.5f;
        bg.transform.localScale = new Vector3(rendered.x + pad, rendered.y + pad, 1f);
        var bounds = tmp.textBounds;
        bg.transform.localPosition = new Vector3(bounds.center.x, bounds.center.y, 0.05f);

        var fdt = go.AddComponent<FloatingDebugText>();
        fdt.tmp = tmp;
        fdt.bgRenderer = bgRend;
    }

    private void Awake()
    {
        var cameraRig = FindFirstObjectByType<OVRCameraRig>();
        cameraTransform = cameraRig != null
            ? cameraRig.centerEyeAnchor
            : Camera.main?.transform;
    }

    private void Update()
    {
        elapsed += Time.deltaTime;
        if (elapsed >= lifetime)
        {
            Destroy(gameObject);
            return;
        }

        // Billboard: face camera
        if (cameraTransform != null)
        {
            transform.LookAt(cameraTransform);
            transform.Rotate(0f, 180f, 0f);
        }

        // Fade out over the last second
        if (elapsed > lifetime - 1f)
        {
            float alpha = Mathf.Clamp01((lifetime - elapsed) / 1f);
            if (tmp != null)
                tmp.color = new Color(tmp.color.r, tmp.color.g, tmp.color.b, alpha);
            if (bgRenderer != null)
                bgRenderer.material.color = new Color(0f, 0f, 0f, alpha);
        }
    }
}
