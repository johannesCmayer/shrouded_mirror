using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AudioManager : MonoBehaviour
{
    public static AudioManager instance;
    AudioSource myAudiosource;

    private void Awake()
    {
        instance = this;
    }

    void Start()
    {
        myAudiosource = GetComponent<AudioSource>();
    }

    public void PlayOneShot2D(AudioClip clip, float volume)
    {
        myAudiosource.PlayOneShot(clip, volume);
    }
}
