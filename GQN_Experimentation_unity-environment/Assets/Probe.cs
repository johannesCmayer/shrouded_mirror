using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Probe : MonoBehaviour
{    
    public AudioSource explodeAudisource;
    public AudioSource signalAudiosource;
    LineRenderer[] lineRenderers;
    public float minPitch = 0.3f;
    public float maxPitch = 3;
    public float pitchCoef = 1;

    bool destroyed;

    // Start is called before the first frame update
    void Start()
    {
        lineRenderers = GetComponentsInChildren<LineRenderer>();
    }

    // Update is called once per frame
    void Update()
    {
        if (destroyed && !explodeAudisource.isPlaying)
            Destroy(gameObject);
        if (destroyed)
        {
            transform.localScale = transform.localScale * 0.99f;
            return;
        }


        Ray forwand = new Ray(transform.position, transform.forward);
        Ray right = new Ray(transform.position, transform.right);
        Ray left = new Ray(transform.position, -transform.right);

        RaycastHit hit;
        if (Physics.Raycast(forwand, out hit))
        {
            signalAudiosource.pitch = Mathf.Max(Mathf.Min((1 / (hit.distance + 0.001f)) * pitchCoef, maxPitch), minPitch);
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        Explode();
    }

    void Explode()
    {
        if (destroyed)
            return;
        GetComponent<MeshRenderer>().enabled = false;
        explodeAudisource.Play();
        signalAudiosource.Stop();
        Destroy(GetComponent<Move>());
        destroyed = true;
    }
}
