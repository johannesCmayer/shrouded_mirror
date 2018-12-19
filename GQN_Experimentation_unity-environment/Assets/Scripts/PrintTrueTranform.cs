using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PrintTrueTranform : MonoBehaviour {

	void Update () {
        print($"{transform.rotation.eulerAngles} rot - {this}");
	}
}
