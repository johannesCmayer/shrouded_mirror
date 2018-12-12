using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DeactivateMeshAtRuntime : MonoBehaviour {
    public bool inChildrenAlso = false;
	void Start () {
        if (inChildrenAlso)
        {
            foreach (var meshRenderer in GetComponentsInChildren<MeshRenderer>())
                meshRenderer.enabled = false;
        }
        else
        {
            GetComponent<MeshRenderer>().enabled = false;
        }
	}
}
