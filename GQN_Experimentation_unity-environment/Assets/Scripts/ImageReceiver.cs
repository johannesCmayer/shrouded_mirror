using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using System.Text;
using UnityEngine.Rendering;
using UnityEngine.Jobs;
using Unity.Jobs;
using Unity.Collections;

public class ImageReceiver : MonoBehaviour {

    public int streamReceivePort = 8686;
    public bool updateNNScreenSizeToMatchStream = true;
    public int nnScreenSizeX = 32;
    public int nnScreenSizeY = 32;

    [Range(0, 255)]
    public int redCutoffValue = 250;
    public Material blitMaterial;

    public Texture2D streamTexture;
    Socket receiver;

    public RenderTexture renderTexture;
    
    bool useJob = false;

    void Start ()
    {
        streamTexture = new Texture2D(nnScreenSizeX, nnScreenSizeY, TextureFormat.RGBA32, false);
        streamTexture.filterMode = FilterMode.Point;

        receiver = Util.GetLocalUDPReceiverSocket(streamReceivePort);
        renderTexture = new RenderTexture(Screen.width, Screen.height, 0);
        renderTexture.filterMode = FilterMode.Point;
        renderTexture.Create();        
	}
    
	void Update ()
    {
        if (receiver.Poll(100, SelectMode.SelectRead))
        {
            byte[] buffer = new byte[2048];
            var msg = receiver.Receive(buffer);
            streamTexture.LoadImage(buffer);
            ProcessStreamTexture(streamTexture);

            if (updateNNScreenSizeToMatchStream)
            {
                nnScreenSizeX = streamTexture.width;
                nnScreenSizeY = streamTexture.width;
            }            
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            useJob = !useJob;
            print($"use job {useJob}");
        }
    }

    void ProcessStreamTexture(Texture2D texture)
    {
        var pix = texture.GetPixels32();
        for (int i = 0; i < pix.Length; i++)
        {
            var item = pix[i];
            var combPixVal = item.a + item.g + item.b;
            if (combPixVal < 10)
                pix[i] = new Color32(255, 255, 255, 0);
            else
                pix[i] = pix[i];
        }
        texture.SetPixels32(pix);
        texture.Apply();
    }

    Texture2D GetUnityEnvTexture(int xSize, int ySize)
    {
        //unityTexture = new Texture2D(xSize, ySize, TextureFormat.RGBA32, false);
        //unityTexture.filterMode = FilterMode.Point;

        Camera.main.targetTexture = renderTexture;
        var unityTexture = new Texture2D(Screen.width, Screen.height);
        unityTexture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        Camera.main.targetTexture = null;

        

        if (useJob)        
            return StartCullRedAndRezizeJob(unityTexture, redCutoffValue, xSize, ySize);
        else
            return ProcessUnityTexture(unityTexture, xSize, ySize);
    }

    Texture2D ProcessUnityTexture(Texture2D toProcess, int newXDim, int newYDim)
    {
        var currentPixels = toProcess.GetPixels32(0);
                
        var width = toProcess.width;
        var height = toProcess.height;
        var newPixels = new Color32[newXDim * newYDim];
        for (int ypos = 0; ypos < newYDim; ypos += 1)
        {
            for (int xpos = 0; xpos < newXDim; xpos++)
            {
                var pixIdx = ypos * width * (height / newYDim) + xpos * (width / newXDim);
                var newPixIdex = ypos * newXDim + xpos;
                var combPixVal = currentPixels[pixIdx].r + currentPixels[pixIdx].g + currentPixels[pixIdx].b;
                if (combPixVal < 1 || 
                    currentPixels[pixIdx].r > redCutoffValue &&
                    currentPixels[pixIdx].g < 10 &&
                    currentPixels[pixIdx].b < 10)
                {
                    newPixels[newPixIdex] = new Color32(0, 255, 0, 0);
                }
                else
                {
                    newPixels[newPixIdex] = currentPixels[pixIdx];
                }
            }
        }
        var newTex = new Texture2D(newXDim, newYDim, TextureFormat.RGBA32, false);
        newTex.filterMode = FilterMode.Point;

        newTex.SetPixels32(newPixels);
        newTex.Apply();
        return newTex;
    }

    private void OnPostRender()
    {
        var unityTex = GetUnityEnvTexture(nnScreenSizeX, nnScreenSizeY);

        Graphics.Blit(streamTexture, blitMaterial);
        Graphics.Blit(unityTex, blitMaterial);
    }

    struct CullRedAndRezizeJob : IJobParallelFor
    {
        public int redCutoffValue;
        public int yDim;
        public int xDim;
        public NativeArray<byte> pixelsR,
                                pixelsG,
                                pixelsB,
                                pixelsA;
        public readonly int Length;

        public void Execute(int i)
        {
            var combPixVal = pixelsR[i] + pixelsG[i] + pixelsB[i];
            if (combPixVal < 1 ||
                pixelsR[i] > redCutoffValue &&
                pixelsG[i] < 10 &&
                pixelsB[i] < 10)
            {
                pixelsR[i] = 0;
                pixelsG[i] = 255;
                pixelsB[i] = 0;
                pixelsA[i] = 0;
            }
            else
            {
                pixelsA[i] = 255;
            }
        }
    }

    Texture2D StartCullRedAndRezizeJob(Texture2D inputTex, int redCutoffValue, int newXDim, int newYDim)
    {
        var pixels = inputTex.GetPixels32();
        var inputTexSize = inputTex.width * inputTex.height;
        var width = inputTex.width;
        var height = inputTex.height;
        var newTexSize = newXDim * newYDim;
        NativeArray<byte> pixelsR = new NativeArray<byte>(newTexSize, Allocator.Temp);
        NativeArray<byte> pixelsG = new NativeArray<byte>(newTexSize, Allocator.Temp);
        NativeArray<byte> pixelsB = new NativeArray<byte>(newTexSize, Allocator.Temp);
        NativeArray<byte> pixelsA = new NativeArray<byte>(newTexSize, Allocator.Temp);

        for (int ypos = 0; ypos < newYDim; ypos += 1)
        {
            for (int xpos = 0; xpos < newXDim; xpos++)
            {
                var pixIdx = ypos * width * (height / newYDim) + xpos * (width / newXDim);
                var newPixIdex = ypos * newXDim + xpos;

                pixelsR[newPixIdex] = pixels[pixIdx].r;
                pixelsG[newPixIdex] = pixels[pixIdx].g;
                pixelsB[newPixIdex] = pixels[pixIdx].b;
                pixelsA[newPixIdex] = pixels[pixIdx].a;
            }
        }

        CullRedAndRezizeJob jobData = new CullRedAndRezizeJob
        {
            redCutoffValue = redCutoffValue,
            yDim = inputTex.height,
            xDim = inputTex.width,
            pixelsR = pixelsR,
            pixelsG = pixelsG,
            pixelsB = pixelsB,
            pixelsA = pixelsA,
        };

        JobHandle handle = jobData.Schedule(newTexSize, 1000);
        handle.Complete();

        var newTex = new Texture2D(newXDim, newYDim);
        var newColors = new Color32[newTexSize];

        for (int i = 0; i < newTexSize; i++)
        {
            newColors[i] = new Color32()
            {
                r = pixelsR[i],
                g = pixelsG[i],
                b = pixelsB[i],
                a = pixelsA[i]
            };
        }

        pixelsR.Dispose();
        pixelsG.Dispose();
        pixelsB.Dispose();
        pixelsA.Dispose();

        newTex.SetPixels32(newColors);
        newTex.Apply();

        return newTex;
    }
}
