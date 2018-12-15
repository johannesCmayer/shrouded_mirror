using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using System.Text;

public class ImageReceiver : MonoBehaviour {

    public int port = 8686;

    public Material matToStreamTo;
    public Material drawTextureMaterial;
    Texture2D streamTexture;
    Socket receiver;

    RenderTexture mainSceneRT;
    RenderTexture streamRenderTexture;

    void Start ()
    {
        streamTexture = new Texture2D(64, 64, TextureFormat.RGBA32, false);
        matToStreamTo.mainTexture = streamTexture;
        receiver = Util.GetLocalUDPReceiverSocket(port);
	}

    
	
	// Update is called once per frame
	void Update ()
    {
        if (receiver.Poll(100, SelectMode.SelectRead))
        {
            byte[] buffer = new byte[2048];
            var msg = receiver.Receive(buffer);
            streamTexture.LoadImage(buffer);
            var pix = streamTexture.GetPixels32();
            var newPix = new Color32[pix.Length];
            for (int i = 0; i < pix.Length; i++)
            {
                var item = pix[i];
                var combPixVal = item.a + item.g + item.b;
                if (combPixVal < 10)
                    newPix[i] = new Color32(255, 255, 255, 0);
                else
                    newPix[i] = pix[i];
            }
            streamTexture.SetPixels32(newPix);
            streamTexture.Apply();
        }
    }

    void OnGUI()
    {
        if (Event.current.type.Equals(EventType.Repaint))
        {
            Graphics.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), streamTexture, drawTextureMaterial);
        }
    }

    void OnPreRender()
    {
        Camera.main.targetTexture = mainSceneRT;
        // this ensures that w/e the camera sees is rendered to the above RT
    }

    void OnPostrender()
    {
        Graphics.Blit(streamTexture, mainSceneRT);
        // You have to set target texture to null for the Blit below to work
        Camera.main.targetTexture = null;
        Graphics.Blit(mainSceneRT, null as RenderTexture);
    }
}
