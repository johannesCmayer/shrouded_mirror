using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ApplyCutAwayMaterial : MonoBehaviour {

    public Material cutAwayMaterial;

	void Start () {
		var allGos = FindObjectsOfType<Renderer>();
        foreach (var renderer in allGos)
        {
            if (LayerMask.LayerToName(renderer.gameObject.layer) != ("NVVisible"))
            {
                renderer.material = cutAwayMaterial;
            }
        }
	}
}
