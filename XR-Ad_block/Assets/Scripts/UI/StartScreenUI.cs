/*
    StartScreenUI

    PURPOSE:
    Drives the initial start screen presented to the user on launch.
    Exposes two buttons — Start and Options — and delegates all
    state transitions to AppStateManager.

    ARCHITECTURE:
    - Start button calls AppStateManager.OnStartPressed which transitions
      to the Running state and enables the pipeline.
    - Options button activates the OptionsUI overlay panel without
      changing application state.
    - This GameObject is activated/deactivated by AppStateManager.SetState.
*/

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
