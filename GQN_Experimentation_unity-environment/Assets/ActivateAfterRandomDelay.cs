using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ActivateAfterRandomDelay : MonoBehaviour
{
    public AudioSource toActivate;

    public float minDelay = 0;
    public float maxDelay = 2;

    // Start is called before the first frame update
    void Start()
    {
        toActivate.enabled = false;
        StartCoroutine(ActivateComponent(Random.Range(minDelay, maxDelay)));
    }

    // Update is called once per frame
    IEnumerator ActivateComponent(float timeToWait)
    {
        yield return new WaitForSeconds(timeToWait);
        toActivate.enabled = true;
    }
}
