using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using System.Net;
using System;

public class StateObject
{
    public Socket socket = null;
    private const int bufferSize = 1024;
    public int BufferSize { get { return bufferSize; } }
    public byte[] buffer = new byte[bufferSize];
    public StringBuilder sb = new StringBuilder();
}

public class DQN_Screen : MonoBehaviour {

    public Texture2D testTex;
    public int port = 5005;

    StateObject state = new StateObject();

    void Start() {
        Thread thread = new Thread(new ThreadStart(ServerSetup));
        thread.Start();
    }

    void ServerSetup()
    {
        IPHostEntry ipHostEntry = Dns.GetHostByName("localhost");
        IPAddress iPAddress = ipHostEntry.AddressList[1];
        IPEndPoint localEndpoint = new IPEndPoint(iPAddress, port);

        Socket socket = new Socket(iPAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp);

        socket.Bind(localEndpoint);
        socket.Listen(1);

        Debug.Log($"Waiting for connection on port {port}");
        state.socket = socket.Accept();
        Debug.Log($"Server connected on port {port}");
    }

    private void Update()
    {
        if (state.socket == null)
            return;

        var clientSocket = state.socket;
        var dataCount = clientSocket.Receive(state.buffer);

        if (dataCount > 0)
        {
            state.sb.Append(Encoding.ASCII.GetString(state.buffer, 0, dataCount));
            string data = state.sb.ToString();
            if (data.IndexOf("<END>") > -1)
            {
                ExtractData(data);
                state = new StateObject();
                state.socket = clientSocket;
            }
        }
        else
        {
            Debug.LogWarning("NO DATA RECEIVED");
        }
    }

    int i = 0;
    void ExtractData(string data)
    {
        i++;
        if (i % 10 == 0)
            print(i / Time.time);
    }

    void UpdateTexture(Texture2D tex)
    {
        if (tex != null)
        {
            var rend = GetComponent<Renderer>();
            rend.material.mainTexture = tex;
        }
        else
        {
            var rend = GetComponent<Renderer>();
            rend.material.mainTexture = testTex;
        }

    }
}
