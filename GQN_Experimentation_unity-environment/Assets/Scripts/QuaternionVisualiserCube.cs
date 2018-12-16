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
        Debug.DrawLine(transform.position, transform.position + new Vector3(myRotation.x, myRotation.y, myRotation.z).normalized * 5, Color.red);
	}
}
