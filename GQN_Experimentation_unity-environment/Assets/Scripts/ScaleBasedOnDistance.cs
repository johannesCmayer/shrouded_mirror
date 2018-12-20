using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScaleBasedOnDistance : MonoBehaviour
{
    public GameObject target;
    public string TagToGetIfNoTarget = "Player";
    public float startScalingAt = 10;
    public float maxSizeAt = 5;

    Vector3 startSize;

    private void Awake()
    {
        startSize = transform.localScale;
    }

    private void Start()
    {
        if (target == null)
        {
            var tagedObjs = GameObject.FindGameObjectsWithTag("Player");
            if (tagedObjs.Length > 1)
                throw new System.Exception("There are multiple objects with that tag");
            if (target == null)
                target = tagedObjs[0];
        }
    }

    private void Update()
    {
        var dist = (target.transform.position - transform.position).magnitude;
        var x = Mathf.Min((startScalingAt - dist) / startScalingAt, 0);
        x = x / (1 - (maxSizeAt / dist));
        x = Mathf.Max(x, 1);
        transform.localScale = startSize * x;
    }
}
