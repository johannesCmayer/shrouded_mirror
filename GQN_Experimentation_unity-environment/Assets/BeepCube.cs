using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BeepCube : MonoBehaviour
{
    AudioSource myAudiosource;
    Renderer myRenderer;
    Color origColor;
    float spectrumMax;

    // Start is called before the first frame update
    void Start()
    {
        myAudiosource = GetComponent<AudioSource>();
        myRenderer = GetComponent<Renderer>();
        origColor = myRenderer.material.color;
    }

    // Update is called once per frame
    void Update()
    {
        var spectrum = new float[64];
        myAudiosource.GetSpectrumData(spectrum, 0, FFTWindow.Rectangular);

        float combinedSpectrum = 0;
        foreach (var item in spectrum)
        {
            combinedSpectrum += item;
        }
        if (combinedSpectrum > spectrumMax)
            spectrumMax = combinedSpectrum;

        var newCol = origColor * (combinedSpectrum / spectrumMax);

        myRenderer.material.color =newCol;

        //for (int i = 1; i < spectrum.Length - 1; i++)
        //{
        //    Debug.DrawLine(new Vector3(i - 1, spectrum[i] + 10, 0), new Vector3(i, spectrum[i + 1] + 10, 0), Color.red);
        //    Debug.DrawLine(new Vector3(i - 1, Mathf.Log(spectrum[i - 1]) + 10, 2), new Vector3(i, Mathf.Log(spectrum[i]) + 10, 2), Color.cyan);
        //    Debug.DrawLine(new Vector3(Mathf.Log(i - 1), spectrum[i - 1] - 10, 1), new Vector3(Mathf.Log(i), spectrum[i] - 10, 1), Color.green);
        //    Debug.DrawLine(new Vector3(Mathf.Log(i - 1), Mathf.Log(spectrum[i - 1]), 3), new Vector3(Mathf.Log(i), Mathf.Log(spectrum[i]), 3), Color.blue);
        //}
    }
}
