/*
    Summary:
    Builds the blocker-image options panel from Resources/BlockerImages and
    stores the selected sprite for future block visuals.

    Pipeline:
    OptionsUI -> BlockerImageSettings -> BlockVisualization
*/

using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class OptionsUI : MonoBehaviour
{
    [SerializeField]
    private Transform imageButtonContainer;

    [SerializeField]
    private GameObject imageButtonPrefab;

    [SerializeField]
    private Button closeButton;

    [SerializeField]
    private Button resetToDefaultButton;

    private List<Sprite> _loadedSprites = new();

    private void Awake()
    {
        closeButton.onClick.AddListener(() => gameObject.SetActive(false));
        resetToDefaultButton.onClick.AddListener(OnResetToDefault);
    }

    private void OnEnable()
    {
        LoadImageOptions();
    }

    private void LoadImageOptions()
    {
        foreach (Transform child in imageButtonContainer)
            Destroy(child.gameObject);

        _loadedSprites.Clear();

        Sprite[] sprites = Resources.LoadAll<Sprite>("BlockerImages");
        Debug.Log($"[Options] Found {sprites.Length} sprites in Resources/BlockerImages");
        foreach (var s in sprites)
            Debug.Log($"[Options] Sprite: {s.name}");
        _loadedSprites.AddRange(sprites);

        foreach (Sprite sprite in _loadedSprites)
        {
            GameObject btn = Instantiate(imageButtonPrefab, imageButtonContainer);
            Debug.Log(
                $"[Options] Instantiated button at position {btn.transform.localPosition}, size {btn.GetComponent<RectTransform>().sizeDelta}"
            );

            Image previewImage = btn.transform.Find("PreviewImage")?.GetComponent<Image>();
            Debug.Log($"[Options] PreviewImage found: {previewImage != null}");

            if (previewImage != null)
            {
                previewImage.sprite = sprite;
                previewImage.preserveAspect = true;
            }
            else
            {
                Debug.LogWarning("[Options] PreviewImage child not found on button prefab");
            }

            TextMeshProUGUI label = btn.GetComponentInChildren<TextMeshProUGUI>();
            if (label != null)
                label.text = sprite.name;
            else
                Debug.LogWarning("[Options] TextMeshProUGUI not found on button prefab");

            Sprite captured = sprite;
            btn.GetComponent<Button>().onClick.AddListener(() => OnImageSelected(captured));
        }

        Canvas.ForceUpdateCanvases();
        UnityEngine.UI.LayoutRebuilder.ForceRebuildLayoutImmediate(
            imageButtonContainer.GetComponent<RectTransform>()
        );
    }

    private void OnImageSelected(Sprite sprite)
    {
        BlockerImageSettings.SelectedSprite = sprite;
        gameObject.SetActive(false);
    }

    private void OnResetToDefault()
    {
        BlockerImageSettings.SelectedSprite = null;
        gameObject.SetActive(false);
    }
}
