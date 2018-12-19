using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ApplyCutAwayMaterial : MonoBehaviour {

    public static ApplyCutAwayMaterial instance;

    public Material cutAwayMaterial;
    bool isActive;
    bool overwriteSetting = false;
    bool manualDeactivate = false;
    bool applyOnStart = true;    
    Renderer[] renderers;
    List<Material> origMaterials = new List<Material>();

    private void Awake()
    {
        instance = this;
        renderers = FindObjectsOfType<MeshRenderer>();
        foreach (var renderer in renderers)
        {

            if (LayerMask.LayerToName(renderer.gameObject.layer) != ("NVVisible"))
            {
                origMaterials.Add(renderer.material);
            }
        }
    }

    public void Deactivate()
    {
        if (!isActive && !applyOnStart)
            return;
        isActive = false;
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
        if (isActive)
            return;
        isActive = true;
        
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

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.C))
        {
            manualDeactivate = !manualDeactivate;
            overwriteSetting = true;
        }
        if (manualDeactivate && overwriteSetting)
            Deactivate();
        else if (overwriteSetting)
            Activate();
    }
}
