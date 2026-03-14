/*
    CroppedImageToTensor

    PURPOSE:
    Converts a cropped ROI texture (from detection stage)
    into a Tensor<float> formatted for OCR model input.

    PIPELINE POSITION:

    Detector → CropPlaceHolder → THIS → OCR Inference

    IMPORTANT:
    - Runs fully on GPU
    - Throttled tensor production, not allocated per cropped image
    - Emits tensor via event
*/
using System;
using Unity.InferenceEngine;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Rendering;

public class CroppedImageToTensor : MonoBehaviour
{

    // PLACEHOLDER
    [SerializeField] CropPlaceHolder croppedTexture;

    // ** Paddle OCR tar in en variable size width. optimal width borde undersökas. 256, 192 är andra förslag**

    [SerializeField] int targetWidth = 320;

    [SerializeField] int targetHeight = 48;

    [SerializeField] private float processingInterval = 0.25f;

    private float lastProcessTime = 0f;

    private RenderTexture renderTexture;

    private CommandBuffer commandBuffer;
    
    /*
        Emits:
        - Tensor<float> (OCR input)
        - FrameData (metadata from original frame)
    */
    public event Action<Tensor<float>, FrameData> sendTensor;

    private void Awake()
    {
        // Allocate GPU render texture
        renderTexture = new RenderTexture(targetWidth,targetHeight,0,RenderTextureFormat.ARGB32);

        renderTexture.Create();

        commandBuffer = new CommandBuffer();
    }

    private void OnEnable()
    {
        if (croppedTexture != null)
        {
            croppedTexture.sendCroppedImage += convertToTensor;
        }
    }

        private void OnDisable()
    {
        if (croppedTexture != null)
        {
            croppedTexture.sendCroppedImage -= convertToTensor;
        }
    }

    /*
        convertToTensor()

        Steps:
        1. Validate texture
        2. Resize ROI
        3. Convert to Tensor
        4. Emit event
    */
    private void convertToTensor(Texture texture, FrameData frame) 
    {

        // Throttle
        if(Time.time - lastProcessTime < processingInterval)
        {
            return;
        }

        lastProcessTime = Time.time;


        if (texture == null)
        {
            Debug.Log("null texture");
            return;
        }

        /*
            OCR model expects:
            (Batch, Channels, Height, Width)
            NCHW format.
        */

        Tensor<float> currentTensor = new Tensor<float>(new TensorShape(1,3,targetHeight, targetWidth));

        commandBuffer.Clear();

        // Resize ROI → target resolution
        commandBuffer.Blit(texture, renderTexture);

        // Convert GPU texture → Tensor
        commandBuffer.ToTensor(renderTexture,currentTensor);

        Graphics.ExecuteCommandBuffer(commandBuffer);

        sendTensor?.Invoke(currentTensor, frame);
    }

        private void OnDestroy()
    {
        if (commandBuffer != null)
        {
            commandBuffer.Release();
            commandBuffer = null;
        }

        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
            renderTexture = null;
        }
    }


}
