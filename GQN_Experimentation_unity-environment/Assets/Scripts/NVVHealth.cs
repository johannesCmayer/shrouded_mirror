using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NVVHealth : MonoBehaviour
{
    public GameObject deathParticalGO;

    private void OnDestroy()
    {
        Instantiate(deathParticalGO, transform.position, Quaternion.identity);
    }
}
