using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Shoot : MonoBehaviour
{
    public GameObject bulletPrefab;
    public float cooldown;
    public int numBulletsSpawned = 10;
    public Vector3 offset = new Vector3(0.4f, 0.4f, 0.4f);
    public Vector3 rotOffset = new Vector3(0.4f, 0.4f, 0.4f);

    void Update()
    {
        if (Input.GetButton("Fire1"))
        {
            for (int i = 0; i < numBulletsSpawned; i++)
            {
                Instantiate(bulletPrefab, transform.position + transform.forward + Vector3.up + RandVec(offset),
                    transform.rotation * Quaternion.Euler(RandVec(rotOffset)));
            }
        }

    }

    Vector3 RandVec(Vector3 offset)
    {
        return new Vector3(Random.Range(-offset.x, offset.x), Random.Range(-offset.y, offset.y), Random.Range(-offset.z, offset.z));
    }
}
