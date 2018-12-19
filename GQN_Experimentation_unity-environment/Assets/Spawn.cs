using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Spawn : MonoBehaviour
{
    public GameObject[] spawnObjects;

    public float minIntervall=1;
    public float maxIntervall = 2;

    float currentIntervall;
    float timer;

    void Start()
    {
        SetIntervall();
    }

    void SetIntervall()
    {
        currentIntervall = Random.Range(minIntervall, maxIntervall);
    }

    void SpawnRandomObject()
    {
        if (spawnObjects.Length != 0)
            Instantiate(spawnObjects[Random.Range(0, spawnObjects.Length)], transform.position, transform.rotation);
    }

    void Update()
    {
        timer += Time.deltaTime;
        if (timer > currentIntervall)
        {
            timer = 0;
            SetIntervall();
            SpawnRandomObject();
        }
    }
}
