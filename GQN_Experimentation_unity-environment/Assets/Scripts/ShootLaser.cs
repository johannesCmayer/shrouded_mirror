using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShootLaser : MonoBehaviour
{
    public GameObject laserPrefab;
    public GameObject impactPrefab;
    public Vector3 offset = Vector3.down * 0.4f;
    public AudioClip[] shootSounds;

    GameObject activeLaser;
    
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetButtonDown("Fire2"))
        {
            StartCoroutine(Shoot());            
        }

    }

    IEnumerator Shoot()
    {        
        var aS = GetComponent<AudioSource>();
        aS.clip = shootSounds[Random.Range(0, shootSounds.Length)];
        aS.Stop();
        aS.Play();
        yield return new WaitForSeconds(0.1f);

        Ray ray = new Ray(transform.position + offset, transform.forward);
        RaycastHit hit;
        if (Physics.Raycast(ray, out hit, 100))
        {
            Instantiate(impactPrefab, hit.point, transform.rotation);
        }
        Instantiate(laserPrefab, transform.position + offset, transform.rotation);
    }
}
