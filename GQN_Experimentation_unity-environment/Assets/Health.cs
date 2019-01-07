using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Health : MonoBehaviour
{    
    Vector3 spawnPosition;

    // Start is called before the first frame update
    void Start()
    {
        spawnPosition = transform.position;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void Die()
    {
        transform.position = spawnPosition;
        var x = gameObject.GetComponent<RigidbodyFirstPersonController>();
    }
}
