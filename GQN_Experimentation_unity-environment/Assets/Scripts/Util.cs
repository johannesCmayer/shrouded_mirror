using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Globalization;
using System.Threading;

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

    public static Socket GetUDPReceiverSocket(int port, string ipAdressString="")
    {
        var receiver = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        receiver.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);

        IPAddress ipAddress = null;
        if (ipAdressString == "")
        {
            IPHostEntry hostEntry = Dns.GetHostEntry(Dns.GetHostName());
            ipAddress = hostEntry.AddressList[1];

        }
        else
        {
            ipAddress = IPAddress.Parse(ipAdressString);
        }

        IPEndPoint endpoint = new IPEndPoint(ipAddress, port);
        receiver.Bind(endpoint);
        return receiver;
    }

    public static float[] RotationEncoding(Quaternion rotation)
    {
        var eulerAngles = rotation.eulerAngles * Mathf.Deg2Rad;
        var data = new[] {
            Mathf.Sin(eulerAngles.x), Mathf.Cos(eulerAngles.x),
            Mathf.Sin(eulerAngles.y), Mathf.Cos(eulerAngles.y),
            Mathf.Sin(eulerAngles.z), Mathf.Cos(eulerAngles.z)
        };
        return data;
    }

    public static Quaternion RotationDecoding(IList list)
    {
        return Quaternion.identity;
    }

    public static string JoinToString(IList list, string seperator = ", ")
    {
        var sb = new StringBuilder();
        for (int i = 0; i < list.Count; i++)
        {
            var dp = list[i];
            sb.Append(dp.ToString());
            if (i < list.Count - 1)
                sb.Append(seperator);
        }
        return sb.ToString();
    }

    public static Vector3 RandVec(Vector3 offset)
    {
        return new Vector3(Random.Range(-offset.x, offset.x), Random.Range(-offset.y, offset.y), Random.Range(-offset.z, offset.z));
    }
}
