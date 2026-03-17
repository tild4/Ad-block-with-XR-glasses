using System;
using JetBrains.Annotations;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public static class ConvertToTensor
{
    public static Tensor<float> convert(Texture texture, RenderTexture renderTexture, int targetHeight, int targetWidth, CommandBuffer commandBuffer)
    {
        if (texture == null)
        {
            return null;
        }

        Tensor<float> tensor = new Tensor<float>(new TensorShape(1, 3, targetHeight, targetWidth));

        commandBuffer.Clear();
        commandBuffer.Blit(texture, renderTexture);
        commandBuffer.ToTensor(renderTexture,tensor);
        Graphics.ExecuteCommandBuffer(commandBuffer);

        Debug.Log("New tensor created!");

        return tensor;
    }

        public static Tensor<float> convert(Texture texture, RenderTexture renderTexture, int targetHeight, int targetWidth, int targetBatchNr, CommandBuffer commandBuffer)
    {
        if (texture == null)
        {
            return null;
        }

        Tensor<float> tensor = new Tensor<float>(new TensorShape(targetBatchNr, 3, targetHeight, targetWidth));

        commandBuffer.Clear();
        commandBuffer.Blit(texture, renderTexture);
        commandBuffer.ToTensor(renderTexture,tensor);
        Graphics.ExecuteCommandBuffer(commandBuffer);

        return tensor;
    }

}
