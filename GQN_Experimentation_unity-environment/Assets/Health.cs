using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.Characters.FirstPerson;

public class Health : MonoBehaviour
{
    public float respawnTime = 2;
    Vector3 spawnPosition;

    bool dead;

    void Start()
    {
        spawnPosition = transform.position;
    }

    private void Update()
    {
        if (dead && Input.anyKeyDown)
            Respawn();
    }

    public void Die()
    {
        UIManager.instance.EnableDeathScreen(true);
        GetComponent<RigidbodyFirstPersonController>().enabled = false;
        dead = true;
    }

    public IEnumerator DelayedRespawn(float delay)
    {
        yield return new WaitForSeconds(delay);
        Respawn();
    }

    public void Respawn()
    {
        transform.position = spawnPosition;
        GetComponent<RigidbodyFirstPersonController>().enabled = true;
        UIManager.instance.EnableDeathScreen(false);
        dead = false;
    }
}
