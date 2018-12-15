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
        var graph = new TFGraph();
        TFOutput input_1 = graph.Placeholder(TFDataType.Float);
        TFOutput input_2 = graph.Placeholder(TFDataType.Float);

        TFOutput z = graph.Add(input_1, input_2);

        TFOutput input, output;

        TFTensor data = new[] { 10, 10 };

        //using (var session = new TFSession(graph))
        //{
        //    var calculatedVal = session.Run(
        //        inputs: new[] { input_1, input_2 },
        //        inputValues: new[] { data },
        //        outputs: new[] { output }
        //        );
        //} 
	}
}
