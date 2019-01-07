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
    Dictionary<Renderer, Material> originals = new Dictionary<Renderer, Material>();

    private void Awake()
    {
        instance = this;
        var renderers = FindObjectsOfType<MeshRenderer>();
        foreach (var renderer in renderers)
        {
            if (LayerMask.LayerToName(renderer.gameObject.layer) != ("NVVisible"))
            {
                originals.Add(renderer, renderer.material);
            }
        }
    }

    public void Deactivate()
    {
        if (!isActive && !applyOnStart)
            return;
        isActive = false;
        applyOnStart = false;
        foreach (var item in originals)
        {
            if (item.Key != null)
                item.Key.material = item.Value;
        }
    }

    public void Activate()
    {
        if (isActive)
            return;
        isActive = true;
        
        foreach (var orig in originals)
        {
            if (LayerMask.LayerToName(orig.Key.gameObject.layer) != ("NVVisible") &&
                orig.Key != null)
            {
                orig.Key.material = cutAwayMaterial;
            }
        }
    }

    void Start ()
    {
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
