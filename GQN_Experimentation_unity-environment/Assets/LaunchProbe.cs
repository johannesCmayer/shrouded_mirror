using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LaunchProbe : MonoBehaviour
{
    public GameObject probePrefab;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            var probe = Instantiate(probePrefab, transform.position, Quaternion.identity);
            probe.transform.forward = transform.forward;
        }
        
    }
}
