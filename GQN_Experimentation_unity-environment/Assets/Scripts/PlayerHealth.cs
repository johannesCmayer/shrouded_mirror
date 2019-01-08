using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.Characters.FirstPerson;

public class PlayerHealth : MonoBehaviour, IKillable
{
    public float respawnTime = 2;
    Vector3 spawnPosition;
    Quaternion spawnRotation;

    bool gameOver;

    void Start()
    {
        spawnPosition = transform.position;
        spawnRotation = transform.rotation;
    }

    private void Update()
    {
        if (gameOver && Input.anyKeyDown)
            Respawn();
    }

    public void Kill()
    {
        TransientPlayer(true);
        UIManager.instance.EnableDeathScreen(true);
        var am = AudioManager.instance;
        am.PlayOneShot2D(am.playerDeath);
        gameOver = true;
    }   

    public void GameWon()
    {
        TransientPlayer(true);
        UIManager.instance.winScreen.SetActive(true);
        var am = AudioManager.instance;
        am.PlayOneShot2D(am.wonGame);
        gameOver = true;
    }

    public void Respawn()
    {
        TransientPlayer(false);
        transform.position = spawnPosition;
        UIManager.instance.EnableDeathScreen(false);
        UIManager.instance.winScreen.SetActive(false);
        gameOver = false;
    }

    public void TransientPlayer(bool enable)
    {
        EnableColliders(!enable);
        GetComponent<RigidbodyFirstPersonController>().enabled = !enable;
    }

    public void EnableColliders(bool enabled)
    {
        foreach (var comp in GetComponentsInChildren<Collider>())
        {
            comp.enabled = enabled;
        }
    }
}
