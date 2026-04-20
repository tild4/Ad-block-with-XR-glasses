/*
    OptionsUI

    PURPOSE:
    Allows the user to select a custom image to display on the blocking
    quad instead of the default transparent blue material. Images are
    loaded at runtime from the Resources/BlockerImages folder.

    ARCHITECTURE:
    - On enable, loads all Sprites from Resources/BlockerImages and
      instantiates one button per image using imageButtonPrefab.
    - Selecting an image stores it in the static BlockerImageSettings
      so BlockVisualization can read it when spawning blocks.
    - Reset to Default clears the selection, restoring the default material.
    - Close button hides the panel without changing the selection.

    SETUP:
    - Place image files inside Assets/Resources/BlockerImages/.
    - Images must be imported as Sprite (2D and UI) texture type.
*/

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
        Debug.Log($"[Options] Found {sprites.Length} sprites in Resources/BlockerImages");
        foreach (var s in sprites)
            Debug.Log($"[Options] Sprite: {s.name}");
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
