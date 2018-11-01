using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class QuaternionVisualiserCube : MonoBehaviour {
    
    public Quaternion myRotation;

	// Use this for initialization
	void Start () {
		
	}
	
	void Update () {
        myRotation = transform.rotation;
        print(transform.rotation);
	}
}
