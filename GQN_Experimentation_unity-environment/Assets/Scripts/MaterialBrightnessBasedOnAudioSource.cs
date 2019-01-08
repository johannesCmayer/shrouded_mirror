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
        var combinedSpectrum = new SpecrumAnalyser(myAudiosource).GetCombinedSpectrum();

        if (combinedSpectrum > spectrumMax)
            spectrumMax = combinedSpectrum;
        if (combinedSpectrum > currentCombinedSpectrum)
            currentCombinedSpectrum = combinedSpectrum;
        else
            currentCombinedSpectrum = Mathf.Lerp(currentCombinedSpectrum, combinedSpectrum, 1f * Time.deltaTime);

        var newCol = origColor * (currentCombinedSpectrum / (spectrumMax / brightnessCoef));

        myRenderer.material.color = newCol;
    }
}
