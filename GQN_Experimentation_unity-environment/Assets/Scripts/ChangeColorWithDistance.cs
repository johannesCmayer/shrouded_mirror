using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChangeColorWithDistance : MonoBehaviour
{
    Renderer[] myRends;
    Color[] origColors;

    ChangeColorWithDistanceGlobalSettings settings;

    void Start()
    {
        settings = ChangeColorWithDistanceGlobalSettings.instance;
        settings.UpdateColors += UpdateColors;

        myRends = GetComponentsInChildren<Renderer>();
        origColors = new Color[myRends.Length];
        for (int i = 0; i < myRends.Length; i++)
            origColors[i] = myRends[i].material.color;
    }

    void UpdateColors()
    {
        if (ModeManager.instance.engineMode == EngineMode.RenderingNetwork)
            return;
        var distTarget = (transform.position -  settings.target.transform.position).magnitude;

        for (int i = 0; i < myRends.Length; i++)
            myRends[i].material.color = origColors[i] / Mathf.Max(settings.maxColorMult, (distTarget * settings.colorDistCoef));
    }

    private void OnDestroy()
    {
        settings.UpdateColors -= UpdateColors;
    }
}
