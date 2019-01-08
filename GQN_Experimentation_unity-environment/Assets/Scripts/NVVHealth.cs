using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NVVHealth : MonoBehaviour, IKillable
{
    public GameObject deathParticalGO;

    private void Kill()
    {
        Instantiate(deathParticalGO, transform.position, Quaternion.identity);
        Destroy(gameObject);
    }
}
