/*
    ProcessOCRDetection2

    PURPOSE:
    Post-processes OCR text-detection output masks into word boxes,
    crops those word ROIs from the original frame, converts them to
    recognition tensors, and forwards them to OCR recognition.

    CURRENT FLOW:
    TextDetectionInference -> THIS -> TextRecognitionInference

    POLICY:
    - Latest batch wins.
    - The class owns incoming tensors and disposes dropped or consumed ones.
    - Processing yields during heavy CPU work to avoid monopolizing the frame.

    NOTE:
    Emitted tensor is the detected text

*/

using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;
public class ProcessOCRDetection_uml : MonoBehaviour
{
    private const int MaskSize = 640;
    private const int MaskYieldStride = 32;
    private const int BfsYieldStride = 5000;
    private const int MinBoxWidth = 10;
    private const int MinBoxHeight = 10;
    private const int PaddingX = 4;
    private const int PaddingY = 2;

    /*
    ============================== UI ==============================
    [SerializeField] private ViewCroppedImage viewCroppedImage;
    private RenderTexture debugPreviewRT;
    ===================================================================
    */

    [SerializeField]
    private TextDetectionInference_uml textDetectionInference;

    // Reused threshold mask for OCR text-detection output.
    private readonly bool[,] mask = new bool[MaskSize, MaskSize];

    [SerializeField]
    private float maskThreshold = 0.3f;

    /*
        OCR recognition model input:
        [DynamicDimension.0, 3, 48, DynamicDimension.1] in NCHW format.

        Width can vary, though multiples of 32 are recommended.
        Batch remains 1 per ROI.
    */
    [SerializeField]
    private int tensorTargetHeight = 48;

    [SerializeField]
    private int tensorTargetWidth = 320;

    /*
    [SerializeField]
    private int cropTargetHeight = 128;

    [SerializeField]
    private int cropTargetWidth = 512;
    */
    

    [SerializeField]
    private Material cropMaterial;

    // Reusable GPU resources
    private RenderTexture convertRenderTexture;
    private RenderTexture croppedROI;
    private CommandBuffer commandBuffer;
    private bool isProcessing = false;

    public event Action<TextTensorsPerAd> sendCroppedROIText;

    private void Awake()
    {
        if (textDetectionInference == null || cropMaterial == null)
        {
            Debug.Log("Missing asset");
            return;
        }

        convertRenderTexture = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        // CHANGE MAYBE
        croppedROI = new RenderTexture(
            tensorTargetWidth, 
            tensorTargetHeight, 
            0,
            RenderTextureFormat.ARGB32
        );

        croppedROI.Create();
        convertRenderTexture.Create();

        commandBuffer = new CommandBuffer();

        /*
        ============================== UI ==============================
        debugPreviewRT = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        debugPreviewRT.Create();
        ===================================================================
        */
    }

    private void OnEnable()
    {
        if (textDetectionInference != null)
        {
            textDetectionInference.findTextRegions += HandleNewTrackedObject;
        }
    }

    private void OnDisable()
    {
        if (textDetectionInference != null)
        {
            textDetectionInference.findTextRegions -= HandleNewTrackedObject;
        }
    }

    /*
        Receives a new OCR text-detection batch.

        FLOW:
        1. Ignore empty input.
        2. Keep only the newest pending batch.
        3. Start processing if needed.
    */
    private void HandleNewTrackedObject(DetectionsPerAd advertisment)
    {
        if (advertisment.trackedObject == null || advertisment.findTextTensor == null)
        {
            return;
        }

        if (!isProcessing)
        {
            StartCoroutine(ProcessDPA(advertisment));
        }
    }


    /*
        Processes batches sequentially.

        POLICY:
        - Finish the current batch.
        - Then process the newest pending batch, if one exists.
    */
    private IEnumerator ProcessDPA(DetectionsPerAd advertisement)
    {
        // Prevents nested coroutines
        isProcessing = true;

        yield return ProcessDetectionOCR(advertisement);

        isProcessing = false;
    }

    /*
        Handles one OCR text-detection tensor.

        FLOW:
        1. Build the threshold mask.
        2. Find connected text boxes.
        3. Dispose the consumed detection tensor.
        4. Crop each text ROI and convert it to a recognition tensor.
        5. Store the frame batch for recognition.
    */
    private IEnumerator ProcessDetectionOCR(DetectionsPerAd advertisement)
    {
        Tensor<float> tensor = advertisement.findTextTensor;
        Rect parentYoloBounds = advertisement.trackedObject.lastDetection.bboxNormalized;
        Texture parentTexture = advertisement.trackedObject.lastDetection.frame.currentTexture;


        if (tensor == null)
        {
            yield break;
        }

        /*
        From "heat map" tensor:
        1. Build text region bounding boxes
        2. Save all boundning boxes
        */

        PipelineProfiler.begin("OCR ProcessBFS");
        yield return BuildMaskFromTensor(tensor);

        List<Rect> boundingBoxes = null;
        yield return FindTextBoxesCoroutine(mask, result => boundingBoxes = result);
        PipelineProfiler.end("OCR ProcessBFS");

        tensor.Dispose();
        
        // Takes the list of bounds and crops the text regions
        List<TextTensor> croppedRois = BuildCroppedRecognitionRois(boundingBoxes, parentYoloBounds, parentTexture);

        TextTensorsPerAd advertisementWithTensors = new TextTensorsPerAd(advertisement.trackedObject, croppedRois);
        sendCroppedROIText?.Invoke(advertisementWithTensors);

    }

    private IEnumerator BuildMaskFromTensor(Tensor<float> tensor)
    {
        for (int y = 0; y < MaskSize; y++)
        {
            for (int x = 0; x < MaskSize; x++)
            {
                mask[y, x] = tensor[0, 0, y, x] > maskThreshold;
            }

            if (y % MaskYieldStride == 0)
            {
                yield return null;
            }
        }
    }

    /*
    NOTE : 
    These bounding boxes are relative to the ad region from YOLO
    Therefore the coordinates need to be converted to be relative
    to the full frame
    */
    private List<TextTensor> BuildCroppedRecognitionRois(List<Rect> boundingBoxes, Rect parentYoloBounds, Texture parentTexture)
    {

        List<TextTensor> croppedRois = new List<TextTensor>();

        /*
        For each detected text region in the ad:
        1. Normalize cooridinates for TextureCropper
        2. Convert the coordinates relative to the ad region to coordinates relative to the full frame
        3. Crop the text region
        4. Convert it to a tensor
        5. Include the relative bounds in emission for debuggning purposes / to view cropped text region
        */

        foreach (Rect bounds in boundingBoxes)
        {

            Rect normalizedLocal = new Rect(
                bounds.x / MaskSize,
                bounds.y / MaskSize,
                bounds.width / MaskSize,
                bounds.height / MaskSize
            );

            Rect normalizedFullFrame = ConvertLocalToFullFrameBounds(normalizedLocal, parentYoloBounds);

            if (!TextureCropper.CropBoundingBox(normalizedFullFrame, parentTexture, croppedROI, cropMaterial))
            {
                continue;
            }

            /*
            ============================== UI ==============================
            Graphics.Blit(croppedROI, debugPreviewRT);
            viewCroppedImage.Show(debugPreviewRT);
            ===================================================================
            */

            // CONVERT WITH ASPECT PAD HERE
            Tensor<float> roiTensor = ConvertToTensor.convert(
                croppedROI,
                convertRenderTexture,
                tensorTargetHeight,
                tensorTargetWidth,
                commandBuffer
            );

            if (roiTensor != null)
            {
                croppedRois.Add(new TextTensor(roiTensor, normalizedFullFrame));
            }
        }

        return croppedRois;
    }

    private Rect ConvertLocalToFullFrameBounds(Rect normalizedLocal, Rect parentYoloBounds)
    {
        return new Rect(
            parentYoloBounds.x + normalizedLocal.x * parentYoloBounds.width,
            parentYoloBounds.y + normalizedLocal.y * parentYoloBounds.height,
            normalizedLocal.width * parentYoloBounds.width,
            normalizedLocal.height * parentYoloBounds.height
        );
    }

    /*
        Connected-component search over the threshold mask.
        Yields periodically so one large mask does not block the main thread.
    */
    private IEnumerator FindTextBoxesCoroutine(bool[,] inputMask, Action<List<Rect>> onComplete)
    {
        int h = inputMask.GetLength(0);
        int w = inputMask.GetLength(1);

        bool[,] visited = new bool[h, w];
        List<Rect> boxes = new List<Rect>();

        int[] dx = { -1, 0, 1, -1, 1, -1, 0, 1 };
        int[] dy = { -1, -1, -1, 0, 0, 1, 1, 1 };

        int workCounter = 0;

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (!inputMask[y, x] || visited[y, x])
                {
                    continue;
                }

                Queue<Vector2Int> queue = new Queue<Vector2Int>();
                queue.Enqueue(new Vector2Int(x, y));
                visited[y, x] = true;

                int minX = x;
                int maxX = x;
                int minY = y;
                int maxY = y;

                while (queue.Count > 0)
                {
                    Vector2Int p = queue.Dequeue();
                    workCounter++;

                    for (int i = 0; i < 8; i++)
                    {
                        int nx = p.x + dx[i];
                        int ny = p.y + dy[i];

                        if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                        {
                            continue;
                        }

                        if (visited[ny, nx] || !inputMask[ny, nx])
                        {
                            continue;
                        }

                        visited[ny, nx] = true;
                        queue.Enqueue(new Vector2Int(nx, ny));

                        if (nx < minX) minX = nx;
                        if (nx > maxX) maxX = nx;
                        if (ny < minY) minY = ny;
                        if (ny > maxY) maxY = ny;
                    }

                    if (workCounter % BfsYieldStride == 0)
                    {
                        yield return null;
                    }
                }

                int width = maxX - minX + 1;
                int height = maxY - minY + 1;

                if (width > MinBoxWidth && height > MinBoxHeight)
                {
                    int paddedMinX = Mathf.Max(0, minX - PaddingX);
                    int paddedMinY = Mathf.Max(0, minY - PaddingY);
                    int paddedMaxX = Mathf.Min(w - 1, maxX + PaddingX);
                    int paddedMaxY = Mathf.Min(h - 1, maxY + PaddingY);

                    boxes.Add(new Rect(
                        paddedMinX,
                        paddedMinY,
                        paddedMaxX - paddedMinX + 1,
                        paddedMaxY - paddedMinY + 1
                    ));
                }
            }

            if (y % MaskYieldStride == 0)
            {
                yield return null;
            }
        }

        onComplete?.Invoke(boxes);
    }

    private void DisposeTensorBatch(List<YoloRoiTensor> batch)
    {
        if (batch == null)
        {
            return;
        }

        foreach (var item in batch)
        {
            item.Tensor?.Dispose();
        }
    }

    private void OnDestroy()
    {

        if (commandBuffer != null)
        {
            commandBuffer.Release();
            commandBuffer = null;
        }

        if (convertRenderTexture != null)
        {
            convertRenderTexture.Release();
            Destroy(convertRenderTexture);
            convertRenderTexture = null;
        }

        if (croppedROI != null)
        {
            croppedROI.Release();
            Destroy(croppedROI);
            croppedROI = null;
        }
        /*
        ============================== UI ==============================
        if (debugPreviewRT != null)
        {
            debugPreviewRT.Release();
            Destroy(debugPreviewRT);
            debugPreviewRT = null;
        }
        ===================================================================
        */
    }
}
