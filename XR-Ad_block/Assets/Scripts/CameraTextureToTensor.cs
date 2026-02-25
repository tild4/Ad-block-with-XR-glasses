using UnityEngine;
using Unity.InferenceEngine;
using UnityEngine.Rendering;
using JetBrains.Annotations;
public class CameraTextureToTensor : MonoBehaviour
{
    private Texture currentTexture;
    [SerializeField] private int targetWidth = 224;
    [SerializeField] private int targetHeight = 224;
    private RenderTexture renderTexture;
    private CommandBuffer commandBuffer;
    public static Tensor<float> currentTensor {get; private set;}

    private void Awake()
    {
        currentTensor = new Tensor<float>(new TensorShape(1, targetHeight, targetWidth, 3));

        renderTexture = new RenderTexture(targetWidth, targetHeight, 0, RenderTextureFormat.ARGB32);

        renderTexture.enableRandomWrite = false;

        renderTexture.Create();

        commandBuffer = new CommandBuffer();
    }

    
    // Update is called once per frame
    private void Update()
    {
        currentTexture = CaptureCameraTexture.currentTexture;

        if (currentTexture == null || renderTexture == null || currentTensor == null)
        {
            return;
        }

        Graphics.Blit(currentTexture, renderTexture);

        commandBuffer.Clear();
        
        commandBuffer.ToTensor(renderTexture,currentTensor);

        Graphics.ExecuteCommandBuffer(commandBuffer);
    }

    private void OnDestroy()
    {
        // Cleanup GPU + native memory
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

        if (currentTensor != null)
        {
            currentTensor.Dispose();
            currentTensor = null;
        }
    }
}
