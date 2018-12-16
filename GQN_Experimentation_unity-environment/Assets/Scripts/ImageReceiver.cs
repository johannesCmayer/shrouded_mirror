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
    Texture2D streamTexture;
    Socket receiver;
    RenderTexture renderTexture;

    void Start ()
    {
        streamTexture = new Texture2D(nnScreenSizeX, nnScreenSizeY, TextureFormat.RGBA32, false);
        streamTexture.filterMode = FilterMode.Point;
        receiver = Util.GetLocalUDPReceiverSocket(streamReceivePort);
        renderTexture = new RenderTexture(nnScreenSizeX, nnScreenSizeY, 0);
        renderTexture.Create();
        
        //Screen.SetResolution(nnScreenSizeX, nnScreenSizeY, false);
	}
    
	void Update ()
    {
        if (receiver.Poll(100, SelectMode.SelectRead))
        {
            byte[] buffer = new byte[2048];
            var msg = receiver.Receive(buffer);
            streamTexture.LoadImage(buffer);
            var pix = streamTexture.GetPixels32();
            for (int i = 0; i < pix.Length; i++)
            {
                var item = pix[i];
                var combPixVal = item.a + item.g + item.b;
                if (combPixVal < 10)
                    pix[i] = new Color32(255, 255, 255, 0);
                else
                    pix[i] = pix[i];
            }
            streamTexture.SetPixels32(pix);
            //if (updateNNScreenSizeToMatchStream && Screen.height != streamTexture.height || Screen.width != streamTexture.width)
            //    Screen.SetResolution(streamTexture.height, streamTexture.width, false);
               
            streamTexture.Apply();
        }
    }

    Texture2D GetUnityEnvTexture()
    {
        var unityEnvTex = new Texture2D(Screen.width, Screen.height, TextureFormat.RGBA32, false);
        unityEnvTex.filterMode = FilterMode.Point;
        unityEnvTex.ReadPixels(new Rect(0, 0, nnScreenSizeX, nnScreenSizeY), 0, 0);

        var pixels = unityEnvTex.GetPixels32();
        for (int i = 0; i < pixels.Length; i++)
        {
            var combPixVal = pixels[i].r + pixels[i].g + pixels[i].b;
            if (combPixVal < 1 || pixels[i].r > redCutoffValue)
                pixels[i] = new Color32(0, 200, 0, 0);
        }
        unityEnvTex.SetPixels32(pixels);
        unityEnvTex.Apply();
        return unityEnvTex;
    }

    private void OnPostRender()
    {
        Camera.main.targetTexture = renderTexture;        
        var unityTex = GetUnityEnvTexture();
        Camera.main.targetTexture = null;


        Graphics.Blit(streamTexture, blitMaterial);
        Graphics.Blit(unityTex, blitMaterial);
    }
}
