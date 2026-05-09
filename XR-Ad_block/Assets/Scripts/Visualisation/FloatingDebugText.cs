using TMPro;
using UnityEngine;

public class FloatingDebugText : MonoBehaviour
{
    private float lifetime = 2.5f;
    private float elapsed;
    private Transform cameraTransform;
    private TextMeshPro tmp;

    public static void Spawn(Vector3 position, TrackedObject obj)
    {
        var go = new GameObject($"FloatingDebug_{obj.id}");
        go.transform.position = position;

        var tmp = go.AddComponent<TextMeshPro>();
        tmp.fontSize = 1.5f;
        tmp.alignment = TextAlignmentOptions.Center;
        tmp.color = Color.red;
        tmp.sortingOrder = 100;
        tmp.richText = true;
        tmp.rectTransform.sizeDelta = new Vector2(4f, 2f);

        const string BG = "<mark=#000000AA padding=\"2,2,2,2\">";
        const string BG_END = "</mark>";

        string shortText = string.IsNullOrEmpty(obj.text) ? "(no text)"
            : (obj.text.Length > 25 ? obj.text.Substring(0, 25) + "..." : obj.text);

        if (obj.nlpScores != null && obj.nlpScores.Length >= 3)
        {
            tmp.text = $"{BG}CHANGED -> UNBLOCKED\n"
                + $"non-ad:{obj.nlpScores[0]:F2} benef:{obj.nlpScores[1]:F2} ad:{obj.nlpScores[2]:F2}\n"
                + $"\"{shortText}\"{BG_END}";
        }
        else
        {
            tmp.text = $"{BG}CHANGED -> UNBLOCKED\n{obj.decisionSource}\n\"{shortText}\"{BG_END}";
        }

        var fdt = go.AddComponent<FloatingDebugText>();
        fdt.tmp = tmp;
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
        if (tmp != null && elapsed > lifetime - 1f)
        {
            float alpha = Mathf.Clamp01((lifetime - elapsed) / 1f);
            tmp.color = new Color(tmp.color.r, tmp.color.g, tmp.color.b, alpha);
        }
    }
}
