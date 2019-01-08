using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class UIManager : MonoBehaviour
{
    public static UIManager instance;
    public GameObject deathScreen;
    public GameObject winScreen;

    void Start()
    {
        instance = this;
    }

    public void EnableDeathScreen(bool enabled)
    {
        deathScreen.SetActive(enabled);
    }

    public void EnableWinScreen(bool enabled)
    {
        winScreen.SetActive(enabled);
    }
}
