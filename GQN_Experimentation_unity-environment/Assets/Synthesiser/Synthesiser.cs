using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Synthesiser : MonoBehaviour
{
    float position;
    float freq = 440;
    double sampleRate;

    private void Start()
    {
        sampleRate = AudioSettings.outputSampleRate;
    }

    private void OnAudioFilterRead(float[] data, int channels)
    {
        for (int i = 0; i < data.Length; i++)
        {
            Mathf.Sin(position * freq);
            position += (float)sampleRate;
        }
    }
}
