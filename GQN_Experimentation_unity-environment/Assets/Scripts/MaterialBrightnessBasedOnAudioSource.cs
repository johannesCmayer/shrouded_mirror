using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(AudioSource))]
public class MaterialBrightnessBasedOnAudioSource : MonoBehaviour
{
    public float brightnessCoef = 2;

    AudioSource myAudiosource;
    Renderer myRenderer;
    Color origColor;
    float spectrumMax;
    float currentCombinedSpectrum;
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
        if (combinedSpectrum > currentCombinedSpectrum)
            currentCombinedSpectrum = combinedSpectrum;
        else
            currentCombinedSpectrum = Mathf.Lerp(currentCombinedSpectrum, combinedSpectrum, 1f * Time.deltaTime);

        var newCol = origColor * (currentCombinedSpectrum / (spectrumMax / brightnessCoef));

        myRenderer.material.color =newCol;
    }
}
