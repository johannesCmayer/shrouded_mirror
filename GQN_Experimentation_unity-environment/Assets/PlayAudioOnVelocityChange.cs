using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayAudioOnVelocityChange : MonoBehaviour
{
    public float volumeScale = 0.001f;
    public AudioSource audiosource;

    private void OnCollisionEnter(Collision collision)
    {
        audiosource.PlayOneShot(audiosource.clip, collision.relativeVelocity.sqrMagnitude * volumeScale);
    }
}
