using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.Characters.FirstPerson;

public class LevelGoal : MonoBehaviour
{
    bool levelFinished;

    private void Update()
    {
        if (levelFinished && Input.anyKeyDown)
        {
            levelFinished = false;
            GameObject.FindGameObjectWithTag("Player").GetComponent<Health>().Respawn();
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player"))
        {            
            UIManager.instance.winScreen.SetActive(true);
            other.gameObject.GetComponent<RigidbodyFirstPersonController>().enabled = false;
            var am = AudioManager.instance;
            am.PlayOneShot2D(am.wonGame);
            Time.timeScale = 0;
            levelFinished = true;
        }
    }
}
