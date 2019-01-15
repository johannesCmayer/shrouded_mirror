using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChangeColorWithDistance : MonoBehaviour
{
    public GameObject target;
    public float colorDistCoef = 0.2f;
    public float maxColorMult = 1;


    Renderer myRend;
    Color origColor;

    void Start()
    {
        myRend = GetComponent<Renderer>();
        origColor = myRend.material.color;
    }

    void Update()
    {
        if (ModeManager.instance.engineMode == EngineMode.RenderingNetwork)
            return;
        var distTarget = (transform.position - target.transform.position).magnitude;
        myRend.material.color = origColor / Mathf.Max(maxColorMult, (distTarget * colorDistCoef));
    }
}
