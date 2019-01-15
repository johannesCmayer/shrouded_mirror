using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShootLaser : MonoBehaviour
{
    public GameObject laserPrefab;
    public GameObject impactPrefab;
    public GameObject noAmmoPrefab;
    public Vector3 offset = Vector3.down * 0.4f;
    public AudioClip shootSound;
    public AudioClip noAmmoSound;

    GameObject activeLaser;

    int ammoCapacity = 3;
    int ammo;
    
    void Start()
    {
        ResetAmmo();
        EventManager.instance.collectedCheckpoint += ResetAmmo;
    }

    void ResetAmmo()
    {
        ammo = ammoCapacity;
        UIManager.instance.SetAmmo(ammo);
    }
    
    void Update()
    {
        if (Input.GetButtonDown("Fire2"))
        {
            if (ammo > 0)
            {
                StartCoroutine(Shoot());
                ammo--;
                UIManager.instance.SetAmmo(ammo);
            }
            else
            {
                StartCoroutine(ShootNoAmmo());
            }
        }
    }

    IEnumerator ShootNoAmmo()
    {
        var aS = GetComponent<AudioSource>();
        aS.clip = noAmmoSound;
        aS.Stop();
        aS.Play();
        yield return new WaitForSeconds(0.1f);

        Instantiate(noAmmoPrefab, transform.position + offset, transform.rotation);
    }

    IEnumerator Shoot()
    {        
        var aS = GetComponent<AudioSource>();
        aS.clip = shootSound;
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
