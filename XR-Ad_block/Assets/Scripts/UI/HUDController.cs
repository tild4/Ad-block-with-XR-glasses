/*
    HUDController

    PURPOSE:
    Manages the heads-up display toast notification shown to the user
    when the blocking pipeline starts. Fades a message in, holds it,
    then fades it out over a configurable duration.

    ARCHITECTURE:
    - Driven by AppStateManager via ShowToast coroutine.
    - Uses TMPro for text rendering.
    - Alpha is animated per-frame using MoveTowards for consistent speed
      regardless of frame rate.
*/

using UnityEngine;
using TMPro;
using System.Collections;

public class HUDController : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI toastText;
    [SerializeField] private float fadeSpeed = 0.7f;

    private Coroutine _activeToast;

    public IEnumerator ShowToast(string message, float duration)
    {
        if (_activeToast != null)
        {
            StopCoroutine(_activeToast);
        }
        _activeToast = StartCoroutine(ToastRoutine(message, duration));
        yield return _activeToast;
    }

    private IEnumerator ToastRoutine(string message, float duration)
    {
        toastText.text = message;

        //Fade in
        float alpha = 0f;
        Color c = toastText.color;
        while (alpha < 1f)
        {
            alpha = Mathf.MoveTowards(alpha, 1f, Time.deltaTime* fadeSpeed);
            toastText.color = new Color(c.r,c.g,c.b, alpha);
            yield return null;
        }

        yield return new WaitForSeconds(duration);

        //Fade out
        while (alpha > 0f)
        {
            alpha = Mathf.MoveTowards(alpha, 0f, Time.deltaTime* fadeSpeed);
            toastText.color = new Color(c.r,c.g,c.b,alpha);
            yield return null;
        }

        toastText.text = "";
        gameObject.SetActive(false); // hide HUD after toast completes
    }
}
