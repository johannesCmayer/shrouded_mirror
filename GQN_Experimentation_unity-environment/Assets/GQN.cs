using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;

public class GQN : MonoBehaviour {

	// Use this for initialization
	void Start () {
        using (var session = new TFSession())
        {
            var graph = session.Graph;

            var a = graph.Const(4);
            var b = graph.Const(54);
            
            var imageInput = graph.Placeholder(TFDataType.Int32);

            var result = session.Run()

            //var addingResult = session.GetRunner().Run(graph.Add(a, b));
            //var addingResultValue = addingResult.GetValue();

            print(result.GetValue());
        }
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
