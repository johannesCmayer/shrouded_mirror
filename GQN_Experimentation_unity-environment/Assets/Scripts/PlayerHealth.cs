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

        UIManager.instance.gameStartScreen.SetActive(true);
    }

    private void Update()
    {
        if (Input.anyKeyDown)
            UIManager.instance.gameStartScreen.SetActive(false);
        if (gameOver && Input.anyKeyDown || Input.GetKeyDown(KeyCode.R))
            Respawn();
    }

    public void Kill()
    {
        TransientPlayer(true);
        UIManager.instance.deathScreen.SetActive(true);
        var am = AudioManager.instance;
        am.PlayOneShot2D(am.playerDeath);
        gameOver = true;
        EventManager.instance.gameOver();
    }   

    public void GameWon()
    {
        TransientPlayer(true);
        UIManager.instance.winScreen.SetActive(true);
        var am = AudioManager.instance;
        am.PlayOneShot2D(am.wonGame);
        gameOver = true;
        EventManager.instance.gameOver();
    }

    public void Respawn()
    {
        TransientPlayer(false);
        transform.position = spawnPosition;
        UIManager.instance.gameStartScreen.SetActive(true);
        UIManager.instance.deathScreen.SetActive(false);
        UIManager.instance.winScreen.SetActive(false);
        gameOver = false;
        EventManager.instance.playerRespawned();
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
