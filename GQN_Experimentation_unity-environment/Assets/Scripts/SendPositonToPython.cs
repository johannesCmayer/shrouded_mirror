using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System.Net;

public class SendPositonToPython : MonoBehaviour {

    public int port = 9797;

    Socket sendSock;
    IPEndPoint endpoint;

	void Start ()
    {        
        sendSock = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        sendSock.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);

        IPHostEntry ipHostEntry = Dns.GetHostByName("localhost");
        IPAddress iPAddress = ipHostEntry.AddressList[1];
        endpoint = new IPEndPoint(iPAddress, port);
    }
	
	// Update is called once per frame
	void Update ()
    {
        var data = transform.position.ToPreciseString() + "_" + transform.rotation.ToPreciseString();
        sendSock.SendTo(Encoding.UTF8.GetBytes(data), endpoint);
    }
}
