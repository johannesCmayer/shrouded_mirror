using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChangeColorWithDistanceGlobalSettings : MonoBehaviour
{
    public static ChangeColorWithDistanceGlobalSettings instance;

    public System.Action UpdateColors = delegate { };

    public GameObject target;
    public float colorDistCoef = 0.2f;
    public float maxColorMult = 1;

    void Awake()
    {
        instance = this;
    }

    private void Start()
    {
    }

    public void UpdateColorsNow()
    {
        UpdateColors();
    }
}
