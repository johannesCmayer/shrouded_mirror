using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class UIManager : MonoBehaviour
{
    public static UIManager instance;
    public GameObject deathScreen;

    void Start()
    {
        instance = this;
    }

    public void EnableDeathScreen(bool enabled)
    {
        deathScreen.SetActive(enabled);
    }
}
