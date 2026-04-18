using UnityEngine;
using UnityEngine.UI;

public class ConfirmStopUI : MonoBehaviour
{
    [SerializeField] private Button yesButton;
    [SerializeField] private Button noButton;

    private void Awake()
    {
        yesButton.onClick.AddListener(() => AppStateManager.Instance.OnConfirmYes());
        noButton.onClick.AddListener(() => AppStateManager.Instance.OnConfirmNo());
    }

}
