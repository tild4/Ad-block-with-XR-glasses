using System.Collections;
using UnityEngine;
using UnityEngine.Android;

public class PermissionManager : MonoBehaviour
{
    private const string HeadsetCameraPermission = "horizonos.permission.HEADSET_CAMERA";
    private const string ScenePermission = "com.oculus.permission.USE_SCENE";

    private bool hasStartedRequestFlow;

    private void Start()
    {
        StartCoroutine(RequestPermissionsWhenReady());
    }

    private IEnumerator RequestPermissionsWhenReady()
    {
        if (hasStartedRequestFlow)
        {
            yield break;
        }

        hasStartedRequestFlow = true;

        while (!Application.isFocused)
        {
            yield return null;
        }

        // Quest can ignore early permission requests if they happen before the app
        // has fully reached the foreground.
        yield return new WaitForSeconds(0.5f);

        RequestIfMissing(HeadsetCameraPermission);
        yield return new WaitForSeconds(0.25f);
        RequestIfMissing(ScenePermission);
    }

    private void RequestIfMissing(string permission)
    {
        bool alreadyGranted = Permission.HasUserAuthorizedPermission(permission);
        Debug.Log($"[Permission] {permission} granted={alreadyGranted}");

        if (alreadyGranted)
        {
            return;
        }

        var callbacks = new PermissionCallbacks();
        callbacks.PermissionGranted += grantedPermission =>
            Debug.Log($"[Permission] Granted: {grantedPermission}");
        callbacks.PermissionDenied += deniedPermission =>
            Debug.LogWarning($"[Permission] Denied: {deniedPermission}");
        callbacks.PermissionDeniedAndDontAskAgain += deniedPermission =>
            Debug.LogError($"[Permission] Denied with don't ask again: {deniedPermission}");
        callbacks.PermissionRequestDismissed += dismissedPermission =>
            Debug.LogWarning($"[Permission] Dismissed: {dismissedPermission}");

        Debug.Log($"[Permission] Requesting: {permission}");
        Permission.RequestUserPermission(permission, callbacks);
    }
}
