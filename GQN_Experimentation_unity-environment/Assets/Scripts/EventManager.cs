using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class EventManager : MonoBehaviour
{
    public static EventManager instance;

    public Action gameOver = delegate { };
    public Action playerRespawned = delegate { };
    public Action collectedCheckpoint = delegate { };

    void Awake()
    {
        instance = this;
    }
}
