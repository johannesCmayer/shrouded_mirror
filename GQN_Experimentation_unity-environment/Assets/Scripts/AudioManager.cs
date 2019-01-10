using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class AudioPlayData
{
    public AudioClip clip;
    public float volume = 1;
}

public class AudioManager : MonoBehaviour
{
    public static AudioManager instance;

    public AudioPlayData playerDeath;
    public AudioPlayData wonGame;
    public AudioPlayData collectedCheckpoint;

    AudioSource myAudiosource;

    private void Awake()
    {
        instance = this;
    }

    void Start()
    {
        myAudiosource = GetComponent<AudioSource>();
    }

    public void PlayOneShot2D(AudioPlayData playData)
    {
        myAudiosource.PlayOneShot(playData.clip, playData.volume);
    }
}
