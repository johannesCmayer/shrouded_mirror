using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ApplyCutAwayMaterial : MonoBehaviour {

    public static ApplyCutAwayMaterial instance;

    public Material cutAwayMaterial;

    bool applyOnStart = true;
    Renderer[] renderers;
    List<Material> origMaterials = new List<Material>();

    private void Awake()
    {
        instance = this;
    }

    public void Deaktivate()
    {
        applyOnStart = false;
        if (renderers == null)
            return;
        for (int i = 0; i < renderers.Length; i++)
        {
             renderers[i].material = origMaterials[i];
        }
    }

    public void Activate()
    {
        renderers = FindObjectsOfType<Renderer>();
        foreach (var renderer in renderers)
        {
            if (LayerMask.LayerToName(renderer.gameObject.layer) != ("NVVisible"))
            {
                origMaterials.Add(renderer.material);
                renderer.material = cutAwayMaterial;
            }
        }
    }
        

    void Start () {
        if (applyOnStart)
        {
            Activate();
        }
	}
}
