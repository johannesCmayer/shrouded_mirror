using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ThrowSmoke : MonoBehaviour
{
    public GameObject prefab;
    public float throwForce = 100;
    public float throwUpCoef = 0.2f;

    void Update()
    {
        if (Input.GetButtonDown("Fire1"))
        {
            var newObj = Instantiate(prefab, transform.position, transform.rotation);
            newObj.GetComponent<Rigidbody>().AddForce((transform.forward + Vector3.up * throwUpCoef) * throwForce);
        }
    }
}
