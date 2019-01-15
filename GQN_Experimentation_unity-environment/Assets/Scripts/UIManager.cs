using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class UIManager : MonoBehaviour
{
    public static UIManager instance;
    public GameObject deathScreen;
    public GameObject winScreen;
    public GameObject gameStartScreen;
    public GameObject ammoDisplay;

    void Start()
    {
        instance = this;
    }

    public void SetAmmo(int count)
    {
        ammoDisplay.GetComponent<TextMeshProUGUI>().text = count.ToString();
    }
}
