// <summary>
// evaluates if an object should be blocked or not based on confindence scores
// from yolo model and OCR

using System;
using System.Collections.Generic;
using System.Linq;
//using System.Diagnostics;
using Meta.XR;
using UnityEngine;

public class DecisionManager : MonoBehaviour
{
    [SerializeField]
    private TextRecognitionInference_uml textRecognitionInference;

    // Event invoked when a decision is made for a tracked object: (obj, concatenatedText, shouldBlock)
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

    // Handle texts produced by TextRecognitionInference pipeline
    public void HandleTexts(TextsPerAd textsPerAd)
    {
        Debug.Log($"[Decision] HandleTexts called for object {textsPerAd.trackedObject?.id}");
        if (textsPerAd.trackedObject == null)
            return;

        // Simple OCR confidence heuristic: presence of any non-empty text -> high confidence
        float ocrConfidence = 0f;
        if (textsPerAd.texts != null && textsPerAd.texts.Count > 0)
        {
            int nonEmpty = textsPerAd.texts.Count(t => !string.IsNullOrWhiteSpace(t));
            ocrConfidence = nonEmpty > 0 ? 0.9f : 0f;
        }

        float yoloConfidence = textsPerAd.trackedObject.lastDetection.confidence;

        var decision = DecisionResult(yoloConfidence, ocrConfidence);

        // Notify subscribers (e.g., TrackingManager) about the decision and provide text
        string combined =
            textsPerAd.texts != null ? string.Join(" ", textsPerAd.texts) : string.Empty;
        onDecisionMade?.Invoke(textsPerAd.trackedObject, combined, decision.shouldBlock);
        Debug.Log(
            $"[Decision] Object {textsPerAd.trackedObject.id} - Final Text: '{combined}', Should Block: {decision.shouldBlock}"
        );
    }

    private (bool shouldBlock, float confidence) DecisionResult(
        float yoloConfidence,
        float ocrConfidence
    )
    {
        // initializing thresholds for blocking
        float upperThreshold = 0.8f;
        float fusionThreshold = 0.7f;
        float lowerThreshold = 0.4f;

        //initializing wheights for confidence fusion
        float yoloWeight = 0.6f;
        float ocrWeight = 0.4f;

        //calculating fusion confidence
        float confidence = yoloConfidence * yoloWeight + ocrConfidence * ocrWeight;

        bool shouldBlock;

        //if yolo confidence good enough, block
        if (yoloConfidence > upperThreshold)
        {
            Debug.Log($"Confidence high: YOLO = {yoloConfidence}, should block = true");
            return (true, yoloConfidence);
        }
        // if yolo model uncertain, use ocr confidence and confidence fusion
        else if (yoloConfidence > lowerThreshold && yoloConfidence < upperThreshold)
        {
            shouldBlock = confidence > fusionThreshold;
            Debug.Log(
                $"Intermediate confidence: fusion confidence = {confidence}, should block:{shouldBlock}"
            );
            return (shouldBlock, confidence);
        }
        // if yolo confidence low, do not block
        else
        {
            Debug.Log($"Confidence low: YOLO = {yoloConfidence}, should block = false");
            return (false, yoloConfidence);
        }
    }

    //lite osäker på om denna del fungerar rätt
    public void Evaluate(float yoloConfidence, float ocrConfidence, Rect rect, FrameData frame)
    {
        var result = DecisionResult(yoloConfidence, ocrConfidence);
        float confidence = result.confidence;
        // This method computes a decision but no longer performs placement.
        // Consumers should subscribe to `onDecisionMade` for full object-level decisions.
    }
}
