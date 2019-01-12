using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Checkpoint : MonoBehaviour
{
    void OnTriggerEnter(Collider c)
    {
        if (c.gameObject.CompareTag("Player"))
        {
            AudioManager.instance.PlayOneShot2D(AudioManager.instance.collectedCheckpoint);
            transform.parent.GetComponent<CheckpointContainer>().CheckpointCollected(gameObject);
        }
    }
}
