using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Jobs;

public class Move : MonoBehaviour
{
    public Vector3 move = Vector3.forward;
    public bool moveToAudisourceOutput;
    public AudioSource audioSource;
    public AudioFilterSettings audioFilterSettings;
    public float minSoundStrengthToMove = 10;
    public bool constantMaxMoveCoeff = true;
    public float maxMoveCoef = 1;

    void Update()
    {
        var scaledMove = move * GetMoveAudiosourceMagnitude() * Time.deltaTime;
        transform.position += 
            transform.forward * scaledMove.z + 
            transform.right * scaledMove.x + 
            transform.up * scaledMove.y;
    }

    float GetMoveAudiosourceMagnitude()
    {
        if (!moveToAudisourceOutput || audioSource == null)
            return 1;

        var combSpectrum = new SpecrumAnalyser(audioSource).GetCombinedSpectrum(audioFilterSettings);
        combSpectrum = Mathf.Pow(combSpectrum, 1);
        if (combSpectrum > minSoundStrengthToMove)
            return Mathf.Min(combSpectrum, maxMoveCoef);
        else
            return 0;
    }
}
