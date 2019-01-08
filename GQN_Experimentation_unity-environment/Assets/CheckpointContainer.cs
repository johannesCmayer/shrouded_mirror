using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CheckpointContainer : MonoBehaviour
{
    List<GameObject> myCheckpoints = new List<GameObject>();
    int currentActiveCheckpoint;

    void Start()
    {
        foreach (Transform trans in transform)
        {
            myCheckpoints.Add(trans.gameObject);
        }
        Reset();
        EventManager.instance.playerRespawned += Reset;
    }
    
    public void CheckpointCollected(GameObject checkpoint)
    {
        myCheckpoints[currentActiveCheckpoint].SetActive(false);
        currentActiveCheckpoint++;
        if (myCheckpoints.Count > currentActiveCheckpoint)
            myCheckpoints[currentActiveCheckpoint].SetActive(true);
    }

    public void Reset()
    {
        currentActiveCheckpoint = 0;
        foreach (var go in myCheckpoints)
        {
            go.gameObject.SetActive(false);
        }
        myCheckpoints[currentActiveCheckpoint].SetActive(true);
    }

    private void OnDestroy()
    {
        EventManager.instance.playerRespawned -= Reset;
    }
}
