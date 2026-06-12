/*
    Summary:
    Converts Unity textures into Tensor<float> inputs for YOLO and OCR
    models, with optional aspect padding and channel swizzling.

    Ownership:
    Callers provide reusable GPU resources and dispose returned tensors.
*/
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public static class ConvertToTensor
{
    public static TextureTransform BgrChannelTransform =>
        new TextureTransform().SetChannelSwizzle(ChannelSwizzle.BGRA);

    private static Rect RecordAspectPadBlit(
        Texture texture,
        RenderTexture renderTexture,
        CommandBuffer commandBuffer
    )
    {
        Rect pixelRect = CalculateAspectFitRect(
            texture.width,
            texture.height,
            renderTexture.width,
            renderTexture.height
        );

        commandBuffer.Clear();
        commandBuffer.SetRenderTarget(renderTexture);
        commandBuffer.ClearRenderTarget(true, true, Color.black);
        commandBuffer.SetViewport(pixelRect);
        commandBuffer.Blit(texture, renderTexture);
        commandBuffer.SetViewport(new Rect(0, 0, renderTexture.width, renderTexture.height));

        return pixelRect;
    }

    public static Rect CalculateAspectFitRect(
        float srcWidth,
        float srcHeight,
        float targetWidth,
        float targetHeight
    )
    {
        float scale = Mathf.Min(targetWidth / srcWidth, targetHeight / srcHeight);
        float scaledWidth = srcWidth * scale;
        float scaledHeight = srcHeight * scale;
        float offsetX = (targetWidth - scaledWidth) * 0.5f;
        float offsetY = (targetHeight - scaledHeight) * 0.5f;
        return new Rect(offsetX, offsetY, scaledWidth, scaledHeight);
    }

    public static Rect BlitWithAspectPad(
        Texture texture,
        RenderTexture renderTexture,
        CommandBuffer commandBuffer
    )
    {
        if (texture == null || renderTexture == null)
        {
            return Rect.zero;
        }

        float srcWidth = texture.width;
        float srcHeight = texture.height;
        if (srcWidth <= 0 || srcHeight <= 0)
        {
            return Rect.zero;
        }

        Rect pixelRect = RecordAspectPadBlit(texture, renderTexture, commandBuffer);
        Graphics.ExecuteCommandBuffer(commandBuffer);
        GL.Flush();

        return new Rect(
            pixelRect.x / renderTexture.width,
            pixelRect.y / renderTexture.height,
            pixelRect.width / renderTexture.width,
            pixelRect.height / renderTexture.height
        );
    }

    public static Tensor<float> convert(
        Texture texture,
        RenderTexture renderTexture,
        int targetHeight,
        int targetWidth,
        CommandBuffer commandBuffer,
        TextureTransform transform = default
    )
    {
        PipelineProfiler.begin("Texture To Tensor (GPU)");
        if (texture == null)
        {
            return null;
        }

        Tensor<float> tensor = new Tensor<float>(new TensorShape(1, 3, targetHeight, targetWidth));

        commandBuffer.Clear();
        commandBuffer.Blit(texture, renderTexture);
        commandBuffer.ToTensor(renderTexture, tensor, transform);

        Graphics.ExecuteCommandBuffer(commandBuffer);
        GL.Flush();

        PipelineProfiler.end("Texture To Tensor (GPU)");

        return tensor;
    }

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

        RecordAspectPadBlit(texture, renderTexture, commandBuffer);

        commandBuffer.ToTensor(renderTexture, tensor);
        Graphics.ExecuteCommandBuffer(commandBuffer);
        GL.Flush();

        PipelineProfiler.end("Texture To Tensor (GPU)");

        return tensor;
    }
}
