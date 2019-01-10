using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class AudioFilterSettings
{
    public float volCutOffLow = 0;
    public float volCutOffHigh = float.MaxValue;
    public float specCutoffLow = 0;
    public float specCutoffHigh = 1;
}

public class SpecrumAnalyser
{
    float[] spectrum;
    readonly int numSamples;

    public SpecrumAnalyser(AudioSource audioSource, FFTWindow fFT = FFTWindow.Rectangular, int numSamples = 64)
    {
        this.numSamples = numSamples;
        spectrum = new float[numSamples];
        audioSource.GetSpectrumData(spectrum, 0, fFT);
    }

    public float[] FilterSpectrum(AudioFilterSettings settings)
    {
        return FilterSpectrum(settings.volCutOffLow, settings.volCutOffHigh, 
            settings.specCutoffLow, settings.specCutoffHigh);
    }

    public float[] FilterSpectrum(
        float volCutOffLow = 0, float volCutOffHigh = float.MaxValue, 
        float specCutoffLow = 0, float specCutoffHigh = 1)
    {
        var newSpectrum = new List<float>();
        var startIdx = (int)(specCutoffLow * numSamples);
        var stopIdx = (int)(spectrum.Length - (1 - specCutoffHigh) * numSamples);
        for (int i = startIdx; i < stopIdx; i++)
        {
            if (spectrum[i] > volCutOffLow && spectrum[i] < volCutOffHigh)
                newSpectrum.Add(spectrum[i]);
        }
        return newSpectrum.ToArray();
    }

    public float GetCombinedSpectrum(AudioFilterSettings settings)
    {
        return GetCombinedSpectrum(settings.volCutOffLow, settings.volCutOffHigh,
            settings.specCutoffLow, settings.specCutoffHigh);
    }

    public float GetCombinedSpectrum(
    float cutOffLow = 0, float cutOffHigh = float.MaxValue,
    float specCutoffLow = 0, float specCutoffHigh = 1)
    {
        float combinedSpectrum = 0;
        foreach (var item in FilterSpectrum(cutOffLow, cutOffHigh, specCutoffLow, specCutoffHigh))
        {
            combinedSpectrum += item;
        }
        return combinedSpectrum;
    }


}
