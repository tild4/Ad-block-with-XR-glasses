/*
    CameraTextureToTensor

    PURPOSE:
    Converts a GPU camera texture into a Tensor<float> that can be used for ML inference.
    
    ARCHITECTURE:
    - Subscribes to CaptureCameraFrame.newFrame event
    - When a new frame arrives:
        1. Rescales it to 224x224
        2. Converts it into a Tensor<float>
        3. Stores it in currentTensor

    IMPORTANT:
    Everything happens on GPU via CommandBuffer.
    No CPU texture readback (fast).
*/
using System;
using JetBrains.Annotations;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public class CameraTextureToTensor : MonoBehaviour
{
    [SerializeField]
    private CaptureCameraFrame cameraFrame;

    [SerializeField]
    private int targetWidth = 640;

    [SerializeField]
    private int targetHeight = 640;

    // GPU texture we render into (resized version of camera frame)
    private RenderTexture renderTexture;

    // GPU command recorder (stores GPU operations before execution)
    private CommandBuffer commandBuffer;

    // Tensor that will hold the image data in (N, C, H, W) format
    private Tensor<float> currentTensor;

    // Event to send tensor
    public event Action<Tensor<float>, FrameData> sendTensor;

    private void Awake()
    {
        currentTensor = new Tensor<float>(new TensorShape(1, 3, targetHeight, targetWidth));

        renderTexture = new RenderTexture(targetWidth, targetHeight, 0, RenderTextureFormat.ARGB32);

        // Allocates GPU memory for it

        renderTexture.Create();

        // CommandBuffer records GPU commands

        commandBuffer = new CommandBuffer();
    }

    /*
        Called every time this component becomes enabled.
        Subscribes to cameraFrame.newFrame here.
        Event-driven architecture.
        No polling.
    */
    private void OnEnable()
    {
        if (cameraFrame != null)
        {
            cameraFrame.newFrame += convertToTensor;
        }
    }

    /*
        Called when component is disabled or destroyed.
        Always unsubscribe from events.
        Prevents memory leaks and null reference errors.
    */

    private void OnDisable()
    {
        if (cameraFrame != null)
        {
            cameraFrame.newFrame -= convertToTensor;
        }
    }

    /*
        This function runs whenever a new camera frame is available.
        Steps:
        1. Validate input
        2. Resize texture (GPU)
        3. Convert texture → Tensor (GPU)
        4. Execute GPU commands
    */

    private void convertToTensor(FrameData frame)
    {
        if (frame.currentTexture == null)
        {
            Debug.Log("null tensor");
            return;
        }

        // Clear previously recorded GPU commands

        commandBuffer.Clear();

        /*
            Blit:
            Copies source texture into renderTexture.
            If sizes differ → automatic GPU scaling.
        */

        commandBuffer.Blit(frame.currentTexture, renderTexture);

        /*
            Converts GPU texture into Tensor<float>.
            This does:
            - Channel extraction
            - Float conversion
            - Layout formatting (NHWC)
            
            NOTE:
            This runs on GPU.
        */

        commandBuffer.ToTensor(renderTexture, currentTensor);

        /*
            Actually sends recorded GPU commands to execute.
            Until this line, commands are only recorded.
        */

        Graphics.ExecuteCommandBuffer(commandBuffer);

        sendTensor?.Invoke(currentTensor,frame);

        Debug.Log("Tensor ready");
    }

    /*
        OnDestroy()
        Called when object is permanently destroyed.

        VERY IMPORTANT:
        GPU memory does NOT get cleaned automatically.
        Must manually release:
        - CommandBuffer
        - RenderTexture
        - Tensor
    */
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

        currentTensor?.Dispose();
        currentTensor = null;
    }
}
