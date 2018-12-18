using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rotate : MonoBehaviour
{
    public Vector3 speed;

    void Update()
    {
        transform.Rotate(speed.x, speed.y, speed.z);
    }
}
