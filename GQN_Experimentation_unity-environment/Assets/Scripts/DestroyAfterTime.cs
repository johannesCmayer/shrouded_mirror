using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DestroyAfterTime : MonoBehaviour
{
    public float timeToDestroy = 10;         
    float timer;

    void Update()
    {
        timer += Time.deltaTime;
        if (timer > timeToDestroy)
            Destroy(gameObject);
    }
}
