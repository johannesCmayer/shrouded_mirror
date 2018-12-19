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

    public Material cutRed;
    public Material pixelate;

    Texture2D streamTexture;
    Socket receiver;

    float timeToLastReceive;

    void Start ()
    {
        streamTexture = new Texture2D(0, 0, TextureFormat.RGBA32, false);
        streamTexture.filterMode = FilterMode.Point;

        receiver = Util.GetLocalUDPReceiverSocket(streamReceivePort);
	}
    
	void Update ()
    {
        UpdateStreamTexture();
    }

    void UpdateStreamTexture()
    {
        if (receiver.Poll(100, SelectMode.SelectRead))
        {
            byte[] buffer = new byte[2048];
            var msg = receiver.Receive(buffer);
            streamTexture.LoadImage(buffer);

            if (updateNNScreenSizeToMatchStream)
            {
                nnScreenSizeX = streamTexture.width;
                nnScreenSizeY = streamTexture.height;
                UpdateBlitMaterial(nnScreenSizeX, nnScreenSizeY);
            }
            timeToLastReceive = 0;
            //blitMaterial.SetInt("_PixelResX", (int)(Mathf.Sin(Time.time) * nnScreenSizeX));
            //blitMaterial.SetInt("_PixelResY", (int)(Mathf.Cos(Time.time) * nnScreenSizeY));
        }
        timeToLastReceive += Time.deltaTime;
    }

    void UpdateBlitMaterial(int newX, int newY)
    {
        pixelate.SetInt("_PixelResX", newX);
        pixelate.SetInt("_PixelResY", newY);
    }

    private void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        if (timeToLastReceive < 2000)
        {
            //ApplyCutAwayMaterial.instance.Activate();

            Graphics.Blit(streamTexture, dest);

            var temp = new RenderTexture(src.width, src.height, 0);
            Graphics.Blit(src, temp, cutRed);
            Graphics.Blit(temp, dest, pixelate);
            temp.Release();
        }
        else
        {
            //ApplyCutAwayMaterial.instance.Deactivate();
            Graphics.Blit(src, dest);            
        }
    }
}
