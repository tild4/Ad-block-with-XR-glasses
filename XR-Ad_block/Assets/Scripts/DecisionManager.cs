// <summary>
 // evaluates if an object should be blocked or not based on confindence scores
 // from yolo model and OCR 

using System.Collections.Generic;
using System.Diagnostics;
using Meta.XR;
using UnityEngine;

public class DecisionManager : MonoBehaviour
{
    public BlockPlacementManager blockPlacementManager;

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
            Debug.log($"Confidence high: YOLO = {yoloConfidence}, should block = true");
            return (true, yoloConfidence);
        }

        // if yolo model uncertain, use ocr confidence and confidence fusion
        else if (yoloConfidence > lowerThreshold && yoloConfidence < upperThreshold)
        {
            shouldBlock = confidence > fusionThreshold;
            Debug.log($"Intermediate confidence: fusion confidence = {confidence}, should block:{shouldBlock}");
            return (shouldBlock, confidence);
        }

        // if yolo confidence low, do not block
        else
        {
            Debug.log($"Confidence low: YOLO = {yoloConfidence}, should block = false" );
            return (false, yoloConfidence);
        }
    }

    //lite osäker på om denna del fungerar rätt
    public void Evaluate(float yoloConfidence, float ocrConfidence, Rect rect, FrameData frame)
    {
        var result = DecisionResult(yoloConfidence, ocrConfidence);
        float confidence = result.confidence;

        if (result.shouldBlock)
        {
            blockPlacementManager.ProcessDetection(rect, confidence, frame);
        }
    }
}
