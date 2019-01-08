﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum EngineMode
{
    DataGeneration,
    RenderingNetwork,
    UnityEnv,
}

public class ModeManager : MonoBehaviour
{
    public static ModeManager instance;
    public EngineMode engineMode;

    public GameObject dataGen;
    public ImageReceiver imageReceiver;
    public GameObject OverlayObjecsRoot;

    void Awake()
    {
        instance = this;

        switch (engineMode)
        {
            case EngineMode.DataGeneration:
                dataGen.SetActive(true);
                imageReceiver.enabled = (false);
                OverlayObjecsRoot.SetActive(false);
                break;
            case EngineMode.RenderingNetwork:
                dataGen.SetActive(false);
                imageReceiver.enabled = (true);
                OverlayObjecsRoot.SetActive(true);
                break;
            case EngineMode.UnityEnv:
                dataGen.SetActive(false);
                imageReceiver.enabled = (false);
                OverlayObjecsRoot.SetActive(true);
                break;
            default:
                break;
        }
    }
}