using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpawnImpactVisuals : MonoBehaviour
{
    public GameObject impactVisuals;

    private void OnTriggerEnter(Collider other)
    {
        var point = other.ClosestPoint(transform.position);
        Instantiate(impactVisuals, point, Quaternion.identity);

    }
}
