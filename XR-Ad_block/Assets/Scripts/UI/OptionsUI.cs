using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using TMPro;

public class OptionsUI : MonoBehaviour
{

    [SerializeField] private Transform imageButtonContainer;
    [SerializeField] private GameObject imageButtonPrefab;
    [SerializeField] private Button closeButton;
    [SerializeField] private Button resetToDefaultButton;

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
        // Clear old buttons
        foreach (Transform child in imageButtonContainer)
            Destroy(child.gameObject);

        _loadedSprites.Clear();

        // Load all sprites from Resources/BlockerImages
        Sprite[] sprites = Resources.LoadAll<Sprite>("BlockerImages");
        _loadedSprites.AddRange(sprites);

        foreach (Sprite sprite in _loadedSprites)
        {
            GameObject btn = Instantiate(imageButtonPrefab, imageButtonContainer);

            // Set preview image
            btn.GetComponentInChildren<Image>().sprite = sprite;

            // Set label to filename
            btn.GetComponentInChildren<TextMeshProUGUI> ().text = sprite.name;

            // Wire selection
            Sprite captured = sprite;
            btn.GetComponent<Button>().onClick.AddListener(() => OnImageSelected(captured));
        }
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
