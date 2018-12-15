﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;

public class Util {

    public Vector3 GetRandomPointInAxisAlignedCube(Transform cubeTransform)
    {
        var cs = cubeTransform.localScale;
        return cubeTransform.transform.position + new Vector3(
            Random.Range(-cs.x / 2, cs.x / 2),
            Random.Range(-cs.y / 2, cs.y / 2),
            Random.Range(-cs.z / 2, cs.z / 2));
    }

    public Vector3 GetRandomPointInAxisAlignedCube(Transform cubeTransform, Vector3 offset)
    {
        return GetRandomPointInAxisAlignedCube(cubeTransform) + offset;
    }

    public static Socket GetLocalUDPReceiverSocket(int port)
    {
        var receiver = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        receiver.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);

        IPHostEntry ipHostEntry = Dns.GetHostByName("localhost");
        IPAddress iPAddress = ipHostEntry.AddressList[1];
        IPEndPoint endpoint = new IPEndPoint(iPAddress, port);

        receiver.Bind(endpoint);
        return receiver;
    }
}
