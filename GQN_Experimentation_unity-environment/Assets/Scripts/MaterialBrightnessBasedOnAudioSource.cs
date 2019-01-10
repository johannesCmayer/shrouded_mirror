using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(AudioSource))]
public class MaterialBrightnessBasedOnAudioSource : MonoBehaviour
{
    public float brightnessCoef = 2;
    public float lerpSpeed = 1;
    public bool setToMaxColor;

    public AudioFilterSettings audioFilterSettings;

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
        myRenderer.material.color = new Color(0, 0, 0, 1);
    }

    // Update is called once per frame
    void Update()
    {
        var combinedSpectrum = new SpecrumAnalyser(myAudiosource).GetCombinedSpectrum(audioFilterSettings);

        if (combinedSpectrum > spectrumMax)
            spectrumMax = combinedSpectrum;
        if (combinedSpectrum > currentCombinedSpectrum)
            currentCombinedSpectrum = combinedSpectrum;
        else
            currentCombinedSpectrum = Mathf.Lerp(currentCombinedSpectrum, combinedSpectrum, lerpSpeed * Time.deltaTime);

        var newCol = origColor * (currentCombinedSpectrum / (spectrumMax / brightnessCoef));
        if (setToMaxColor)
        {
            if (currentCombinedSpectrum > 0)
                newCol = origColor * brightnessCoef;
            else
            {
                newCol = origColor * 0;
            }
        }
        
        newCol.a = 1;
        myRenderer.material.color = newCol;
    }
}
