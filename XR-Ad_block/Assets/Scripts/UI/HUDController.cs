using UnityEngine;
using TMPro;
using Systems.Collections;

public class HUDController : MonoBehaviour
{
    [SerializeField] private TextMeshProGUI toastText;
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
    }
}
