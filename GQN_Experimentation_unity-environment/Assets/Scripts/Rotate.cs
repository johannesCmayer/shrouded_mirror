using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rotate : MonoBehaviour
{
    public Vector3 speed;

    void Update()
    {
        var tempSpeed = speed * Time.deltaTime;
        transform.Rotate(tempSpeed.x, tempSpeed.y, tempSpeed.z);
    }
}
