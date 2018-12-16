using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using System.Text;
using UnityEngine.Rendering;

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

        return ProcessUnityTexture(unityTexture, xSize, ySize);
    }

    Color32 NextCol(int i)
    {
        return new Color32((byte)(i % 255), 0, 0, 255);
    }
    int i = 0;
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
                var pixIdx = ypos * newXDim * (width / newXDim) + xpos * (height / newYDim);
                var newPixIdex = ypos * newXDim + xpos;
                var combPixVal = currentPixels[pixIdx].r + currentPixels[pixIdx].g + currentPixels[pixIdx].b;
                if (combPixVal < 1 || currentPixels[pixIdx].r > redCutoffValue)
                {
                    newPixels[newPixIdex] = new Color32(0, 255, 0, 0);
                }
                else
                {
                    newPixels[newPixIdex] = currentPixels[pixIdx];
                }
                //print(pixIdx);
                //print(newPixIdex);
            }
        }
        var newTex = new Texture2D(newXDim, newYDim, TextureFormat.RGBA32, false);
        newTex.filterMode = FilterMode.Point;

        newTex.SetPixels32(newPixels);
        newTex.Apply();
        return newTex;
    }

    private void OnPreRender()
    {
        
        renderTexture.filterMode = FilterMode.Point;
    }


    private void OnPostRender()
    {
        var unityTex = GetUnityEnvTexture(nnScreenSizeX, nnScreenSizeY);

        Graphics.Blit(streamTexture, blitMaterial);
        Graphics.Blit(unityTex, blitMaterial);
    }
}
