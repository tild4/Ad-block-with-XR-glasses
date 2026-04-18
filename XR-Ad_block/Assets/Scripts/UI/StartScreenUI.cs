using UnityEngine;
using UnityEngine.UI;

public class StartScreenUI : MonoBehaviour
{
    [SerializeField] private Button startButton;
    [SerializeField] private Button optionsButton;
    [SerializeField] private OptionsUI optionsUI;

    private void Awake()
    {
        startButton.onClick.AddListener(OnStart);
        optionsButton.onClick.AddListener(OnOptions);
    }

    private void OnStart()
    {
        AppStateManager.Instance.OnStartPressed();
    }

    private void OnOptions()
    {
        optionsUI.gameObject.SetActive(true);
    }
}
