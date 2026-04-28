/*
    ConfirmStopUI

    PURPOSE:
    Displays a confirmation dialog when the user presses the B button
    during an active blocking session, asking whether they want to stop.

    ARCHITECTURE:
    - Yes button calls AppStateManager.OnConfirmYes which clears all
      active blocks and returns to the start screen.
    - No button calls AppStateManager.OnConfirmNo which resumes the
      running state without any pipeline interruption.
    - This GameObject is activated/deactivated by AppStateManager.SetState.
*/

using UnityEngine;
using UnityEngine.UI;

public class ConfirmStopUI : MonoBehaviour
{
    [SerializeField]
    private Button yesButton;

    [SerializeField]
    private Button noButton;

    private void Awake()
    {
        yesButton.onClick.AddListener(() => AppStateManager.Instance.OnConfirmYes());
        noButton.onClick.AddListener(() => AppStateManager.Instance.OnConfirmNo());
    }
}
