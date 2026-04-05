using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ViewCroppedImage : MonoBehaviour
{
    [SerializeField] private RawImage debugImage;

    [SerializeField] private TextMeshProUGUI wordText;

    public void Show(Texture texture)
    {
        if (debugImage == null)
        {
            Debug.LogWarning("CropDebugUI: debugImage is not assigned.");
            return;
        }

        debugImage.texture = texture;
        debugImage.enabled = texture != null;
    }

    public void SetDetectedWord(string word)
    {
        wordText.text = string.IsNullOrEmpty(word) ? "" : word;
    }

    public void Clear()
    {
        if (debugImage == null)
        {
            return;
        }

        debugImage.texture = null;
        debugImage.enabled = false;
    }
}
