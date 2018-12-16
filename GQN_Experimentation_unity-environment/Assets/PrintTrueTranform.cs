using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PrintTrueTranform : MonoBehaviour {

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
        print($"{transform.rotation.eulerAngles} rot - {this}");
	}
}
