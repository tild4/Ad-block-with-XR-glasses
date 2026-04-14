/*
    ConvertToTensor (Utility Class)

    PURPOSE:
    Provides reusable static methods to convert a Texture → Tensor<float>
    using GPU operations (CommandBuffer + RenderTexture).

    Caller owns:
        - RenderTexture
        - CommandBuffer
        - Tensor disposal

    IMPORTANT:
    This function ALLOCATES a new Tensor every call.
    The caller MUST dispose it when done.
*/
using System;
using JetBrains.Annotations;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public static class ConvertToTensor
{
    /*
    Converts a texture into a tensor with batch size = 1.

    PARAMETERS:
    - texture: input GPU texture (camera frame or cropped image)
    - renderTexture: preallocated GPU texture for resizing
    - targetHeight / targetWidth: model input size
    - commandBuffer: GPU command recorder

    RETURNS:
    - Tensor<float> in NCHW format (1, 3, H, W)
    */
    public static Tensor<float> convert(
        Texture texture,
        RenderTexture renderTexture,
        int targetHeight,
        int targetWidth,
        CommandBuffer commandBuffer
    )
    {
        PipelineProfiler.begin("Texture To Tensor (GPU)");
        if (texture == null)
        {
            return null;
        }

        /*
        Allocate tensor:
        Shape = (Batch, Channels, Height, Width)
        Channels = 3 (RGB)
        */

        Tensor<float> tensor = new Tensor<float>(new TensorShape(1, 3, targetHeight, targetWidth));

        // Clear previously recorded GPU commands
        commandBuffer.Clear();

        // Blit: Copies + resizes texture into renderTexture
        commandBuffer.Blit(texture, renderTexture);

        /*
        Converts GPU texture → Tensor<float>
        Handles:
        - Pixel → float conversion
        - Channel extraction
        - Layout formatting
        */
        commandBuffer.ToTensor(renderTexture, tensor);

        // Execute all recorded GPU commands
        Graphics.ExecuteCommandBuffer(commandBuffer);
            GL.Flush();

        Debug.Log("New tensor created!");

        PipelineProfiler.end("Texture To Tensor (GPU)");

        return tensor;
    }

    // Same as above, but allows custom batch size.
    public static Tensor<float> convert(
        Texture texture,
        RenderTexture renderTexture,
        int targetHeight,
        int targetWidth,
        int targetBatchNr,
        CommandBuffer commandBuffer
    )
    {
        if (texture == null)
        {
            return null;
        }

        Tensor<float> tensor = new Tensor<float>(
            new TensorShape(targetBatchNr, 3, targetHeight, targetWidth)
        );

        commandBuffer.Clear();
        commandBuffer.Blit(texture, renderTexture);
        commandBuffer.ToTensor(renderTexture, tensor);
        Graphics.ExecuteCommandBuffer(commandBuffer);
            GL.Flush();

        return tensor;
    }

    /*
    Converts a texture into a tensor while preserving aspect ratio.
    The source is resized to fit inside the target size, then padded.

    Best for OCR recognition, where character distortion hurts accuracy.

    PARAMETERS:
    - texture: input GPU texture
    - renderTexture: preallocated target RT with final model size
    - targetHeight / targetWidth: model input size
    - commandBuffer: reusable GPU command buffer

    RETURNS:
    - Tensor<float> in NCHW format (1, 3, H, W)
    */
    public static Tensor<float> convertWithAspectPad(
        Texture texture,
        RenderTexture renderTexture,
        int targetHeight,
        int targetWidth,
        CommandBuffer commandBuffer
    )
    {
        PipelineProfiler.begin("Texture To Tensor (GPU)");

        if (texture == null)
        {
            return null;
        }

        Tensor<float> tensor = new Tensor<float>(new TensorShape(1, 3, targetHeight, targetWidth));

        float srcWidth = texture.width;
        float srcHeight = texture.height;

        if (srcWidth <= 0 || srcHeight <= 0)
        {
            tensor.Dispose();
            return null;
        }

        // Scale source to fit inside target while preserving aspect ratio
        float scale = Mathf.Min(targetWidth / srcWidth, targetHeight / srcHeight);

        float scaledWidth = srcWidth * scale;
        float scaledHeight = srcHeight * scale;

        // Center the resized image inside the target
        float offsetX = (targetWidth - scaledWidth) * 0.5f;
        float offsetY = (targetHeight - scaledHeight) * 0.5f;

        commandBuffer.Clear();

        // Clear target to black so remaining area becomes padding
        commandBuffer.SetRenderTarget(renderTexture);
        commandBuffer.ClearRenderTarget(true, true, Color.black);

        // Draw source texture into centered padded rect
        Rect pixelRect = new Rect(offsetX, offsetY, scaledWidth, scaledHeight);
        commandBuffer.SetViewport(pixelRect);
        commandBuffer.Blit(texture, renderTexture);

        // Restore full viewport before tensor conversion
        commandBuffer.SetViewport(new Rect(0, 0, targetWidth, targetHeight));

        commandBuffer.ToTensor(renderTexture, tensor);
        Graphics.ExecuteCommandBuffer(commandBuffer);
            GL.Flush();

        PipelineProfiler.end("Texture To Tensor (GPU)");

        return tensor;
    }
}
