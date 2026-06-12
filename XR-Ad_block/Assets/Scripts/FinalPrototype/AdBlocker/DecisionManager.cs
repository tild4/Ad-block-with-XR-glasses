/*
    Summary:
    Classifies recognized OCR text and emits the final block/no-block
    decision for each tracked object.

    Pipeline:
    TextRecognitionInference -> DecisionManager -> TrackingManager
*/

using System;
using UnityEngine;

public class DecisionManager : MonoBehaviour
{
    [SerializeField]
    private TextRecognitionInference textRecognitionInference;

    [SerializeField]
    private NLPClassifier nlpClassifier;

    [SerializeField, Range(0f, 1f)]
    private float nlpThreshold = 0.5f;

    public event Action<TrackedObject, string, bool> onDecisionMade;

    private void OnEnable()
    {
        if (textRecognitionInference != null)
        {
            textRecognitionInference.sendTexts += HandleTexts;
        }
    }

    private void OnDisable()
    {
        if (textRecognitionInference != null)
        {
            textRecognitionInference.sendTexts -= HandleTexts;
        }
    }

    public void HandleTexts(TextsPerAd textsPerAd)
    {
        Debug.Log($"[Decision] HandleTexts called for object {textsPerAd.trackedObject?.id}");
        if (textsPerAd.trackedObject == null)
            return;

        string combined =
            textsPerAd.texts != null ? string.Join(" ", textsPerAd.texts) : string.Empty;

        nlpClassifier.Classify(
            combined,
            (label, probs) =>
            {
                float ocrConfidence = probs[0] + probs[1];

                var decision = DecisionResult(ocrConfidence, combined);
                onDecisionMade?.Invoke(textsPerAd.trackedObject, combined, decision.shouldBlock);
            }
        );
    }

    private (bool shouldBlock, float confidence) DecisionResult(float ocrConfidence, string text)
    {
        bool shouldBlock = ocrConfidence < nlpThreshold;
        Debug.Log(
            $"[Decision] Evaluating: '{text}' with P(non-ad+beneficial)={ocrConfidence:F2} -> shouldBlock={shouldBlock}"
        );
        return (shouldBlock, ocrConfidence);
    }
}
