using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OffsetPlayTime : MonoBehaviour
{
    AudioSource toOffset;

    public float minStartTime = 0;
    public float maxStartTime = 2;

    void Start()
    {
        toOffset = GetComponent<AudioSource>();
        toOffset.time = Random.Range(minStartTime, maxStartTime);
    }
}
