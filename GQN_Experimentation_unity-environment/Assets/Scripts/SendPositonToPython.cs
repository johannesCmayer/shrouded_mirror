﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System.Net;
using System.Linq;
using System;
using System.Globalization;

public class SendPositonToPython : MonoBehaviour {

    public int port = 9797;

    public string readPosRotFromThis;
    string prevReadPosRotFromThis;

    public string sendDataPos;
    public string sendDataRot;

    Socket sendSock;
    IPEndPoint endpoint;

	void Start ()
    {
        if (ModeManager.instance.engineMode != EngineMode.RenderingNetwork)
            return;
        sendSock = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        sendSock.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);

        IPHostEntry hostEntry = Dns.GetHostEntry("localhost");
        IPAddress ipAddress = hostEntry.AddressList[1];
        endpoint = new IPEndPoint(ipAddress, port);
    }

    void Update()
    {
        if (ModeManager.instance.engineMode != EngineMode.RenderingNetwork)
            return;
        if (prevReadPosRotFromThis != readPosRotFromThis && readPosRotFromThis != "")
        {
            var posRot = readPosRotFromThis.Split('_');
            var pos = posRot[0].Split(new[] { ", " }, StringSplitOptions.None)
                .Select(x => float.Parse(x, CultureInfo.InvariantCulture.NumberFormat)).ToArray().ArrayToVector3();
            var rot = posRot[1].Split(new[] { ", " }, StringSplitOptions.None)
                .Select(x => float.Parse(x, CultureInfo.InvariantCulture.NumberFormat));
        }

        sendDataPos = transform.position.ToPreciseString();
        sendDataRot = Util.JoinToString(Util.RotationEncoding(transform.rotation));

        var sendData = Encoding.UTF8.GetBytes(sendDataPos + "_" + sendDataRot);
        sendSock.SendTo(sendData, endpoint);
    }

    
}
