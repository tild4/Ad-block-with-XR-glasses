/*
    Summary:
    Coordinates app state between the start menu, running ad-blocking
    pipeline, and stop confirmation dialog.

    Pipeline:
    UI buttons and controller input -> AppStateManager -> pipeline component
    enablement and block cleanup.
*/
using System.Collections;
using UnityEngine;

public class AppStateManager : MonoBehaviour
{
    public enum AppState
    {
        StartScreen,
        Running,
        ConfirmStop,
    }

    [Header("Pipeline Components")]
    [SerializeField]
    private CaptureCameraFrame captureCameraFrame;

    [SerializeField]
    private YOLOInferenceManager yoloInferenceManager;

    [SerializeField]
    private BlockPlacementManager blockPlacementManager;

    [Header("UI Controllers")]
    [SerializeField]
    private StartScreenUI startScreenUI;

    [SerializeField]
    private ConfirmStopUI confirmStopUI;

    [SerializeField]
    private HUDController hudController;

    [Header("Controller")]
    [SerializeField]
    private ControllerRay controllerRay;

    public static AppStateManager Instance { get; private set; }
    public AppState CurrentState { get; private set; } = AppState.StartScreen;

    public void Awake()
    {
        Instance = this;
    }

    private void Start()
    {
        Debug.Log("[AppState] Start called, setting StartScreen state");
        SetState(AppState.StartScreen);
    }

    private void Update()
    {
        if (CurrentState == AppState.Running)
        {
            if (
                OVRInput.GetDown(OVRInput.Button.Two)
                || OVRInput.GetDown(OVRInput.Button.SecondaryThumbstick)
            )
            {
                SetState(AppState.ConfirmStop);
            }
        }
    }

    public void SetState(AppState newState)
    {
        Debug.Log($"[AppState] SetState called: {newState}");
        CurrentState = newState;

        bool pipelineActive = newState == AppState.Running;
        Debug.Log($"[AppState] Setting pipeline active: {pipelineActive}");
        captureCameraFrame.enabled = pipelineActive;
        yoloInferenceManager.enabled = pipelineActive;
        blockPlacementManager.enabled = pipelineActive;

        startScreenUI.gameObject.SetActive(newState == AppState.StartScreen);
        confirmStopUI.gameObject.SetActive(newState == AppState.ConfirmStop);

        if (controllerRay != null)
        {
            controllerRay.enabled = newState != AppState.Running;
        }

        if (newState == AppState.Running)
        {
            StartCoroutine(ShowRayDuringToast());
        }
    }

    private IEnumerator ShowRayDuringToast()
    {
        if (controllerRay != null)
        {
            controllerRay.enabled = true;
        }

        hudController.gameObject.SetActive(true);
        yield return StartCoroutine(
            hudController.ShowToast("Press B any time to stop blocking", 4f)
        );

        if (controllerRay != null)
        {
            Debug.Log("[AppState] Hiding controller ray after toast");
            controllerRay.enabled = false;
        }
    }

    public void OnStartPressed() => SetState(AppState.Running);

    public void OnConfirmYes()
    {
        blockPlacementManager.ClearAllBlocks();
        SetState(AppState.StartScreen);
    }

    public void OnConfirmNo() => SetState(AppState.Running);
}
