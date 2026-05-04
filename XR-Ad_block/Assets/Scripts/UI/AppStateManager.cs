/*
    AppStateManager

    PURPOSE:
    Central state machine for the application. Controls transitions between
    the start screen, active blocking pipeline, and the confirm-stop dialog.
    Enables and disables pipeline MonoBehaviours based on the current state
    so no camera or inference work happens when the app is on the start screen.

    STATES:
    - StartScreen : Pipeline disabled. Start/Options UI visible.
    - Running     : Pipeline active. B button monitored for stop request.
    - ConfirmStop : Pipeline paused. Stop confirmation dialog visible.

    ARCHITECTURE:
    - Singleton via Instance property, referenced by all UI scripts.
    - Pipeline components are toggled via .enabled rather than
      GameObject.SetActive so their state is preserved between sessions.
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
    private SentisInferenceManager sentisInferenceManager;

    [SerializeField]
    private BlockPlacementManager_MVP2 blockPlacementManager;

    [Header("UI Controllers")]
    [SerializeField]
    private StartScreenUI startScreenUI;

    [SerializeField]
    private ConfirmStopUI confirmStopUI;

    [SerializeField]
    private HUDController hudController;

    [Header("Controller")]
    [SerializeField] private ControllerRay controllerRay;

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
        // B button only active while running
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

        //Enable pipeline if user has presssed start, app is running.
        bool pipelineActive = newState == AppState.Running;
        Debug.Log($"[AppState] Setting pipeline active: {pipelineActive}");
        captureCameraFrame.enabled = pipelineActive;
        sentisInferenceManager.enabled = pipelineActive;
        blockPlacementManager.enabled = pipelineActive;

        //UI
        startScreenUI.gameObject.SetActive(newState == AppState.StartScreen);
        confirmStopUI.gameObject.SetActive(newState == AppState.ConfirmStop);

        // Ray visible in menus and confirm dialouge, hidden while running
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
        // Show ray while toast is visible
        if (controllerRay != null)
        {
            controllerRay.enabled = true;
        }

        hudController.gameObject.SetActive(true);
        yield return StartCoroutine(hudController.ShowToast("Press B any time to stop blocking", 4f));

        // Hide ray after toast completes
        if (controllerRay != null)
        {
            controllerRay.enabled = false;
        }
    }

    // Called by StartScreenUI Start button
    public void OnStartPressed() => SetState(AppState.Running);

    // Called by ConfirmStopUI
    public void OnConfirmYes()
    {
        blockPlacementManager.ClearAllBlocks();
        SetState(AppState.StartScreen);
    }

    public void OnConfirmNo() => SetState(AppState.Running);
}
