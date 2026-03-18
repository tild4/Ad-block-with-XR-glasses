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

        Debug.Log("New tensor created!");

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

        return tensor;
    }
}
