using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;

public class TFBytesRunner : MonoBehaviour {

    public TextAsset bytesGraph;

	void Start ()
    {
        var graph = new TFGraph();
        graph.Import(bytesGraph.bytes);
        var session = new TFSession(graph);
        var runner = session.GetRunner();
        runner.Run()[0].GetValue();
	}
	
	void Update ()
    {
		
	}
}
