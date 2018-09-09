using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DeactivateMeshAtRuntime : MonoBehaviour {

	void Start () {
        foreach (var meshRenderer in GetComponentsInChildren<MeshRenderer>())
            meshRenderer.enabled = false;
	}
}
