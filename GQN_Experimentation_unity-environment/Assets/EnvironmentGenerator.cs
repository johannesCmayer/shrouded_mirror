using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentGenerator : MonoBehaviour {

    [Header("Setup")]
    public TakeObservation takeObservation;
    public Transform environmentalObjectsSpawnVolume;
    public int capturesBevoreRandomization = 100;
    public int maxNumEnvObjects = 1;
    public int minNumEnvObjects = 1;
    public float environmentObjectsMinScale = 0.4f;
    public float environmentObjectsMaxScale = 2f;
    public GameObject[] environmentBaseObjects;
    public GameObject[] environmentalObjectsPrefabs;

    List<GameObject> environmentalObjects = new List<GameObject>();
    int captureCounter;

	void Start () {
        takeObservation.TookObservation += OnTookObservation;
        RandomizeEnv();
	}

    void OnTookObservation()
    {
        captureCounter++;
        if (captureCounter >= capturesBevoreRandomization)
        {
            captureCounter = 0;
            RandomizeEnv();
        }
    }

    Vector3 GetRandomVec3(float rangeMin=0, float rangeMax=1)
    {
        System.Func<float> r_val = () => Random.Range(rangeMin, rangeMax);
        return new Vector3(r_val(), r_val(), r_val());
    }

    Color GetRandomColor(float rangeMin = 0, float rangeMax = 1)
    {
        System.Func<float> r_val = () => Random.Range(rangeMin, rangeMax);
        return new Color(r_val(), r_val(), r_val());
    }

    void SetRandomColor(GameObject go)
    {
        go.GetComponent<Renderer>().material.color = GetRandomColor();
    }

    void RandomizeEnv()
    {
        print("Randomize Environment");
        takeObservation.cam.backgroundColor = GetRandomColor();
        foreach (var item in environmentBaseObjects)
        {
            SetRandomColor(item);
        }
        foreach (var item in environmentalObjectsPrefabs)
        {
            for (int i = environmentalObjects.Count - 1; i >= 0; i--)
            {
                Destroy(environmentalObjects[i]);
            }
            var envObjs = new List<GameObject>();
            for (int i = 0; i < Random.Range(minNumEnvObjects, maxNumEnvObjects + 1); i++)
            {
                var idx = Random.Range(0, environmentalObjectsPrefabs.Length);
                var newEnvObj = Instantiate(environmentalObjectsPrefabs[idx], 
                                            new Util().GetRandomPointInAxisAlignedCube(environmentalObjectsSpawnVolume),
                                            Random.rotation);
                newEnvObj.transform.localScale = GetRandomVec3(environmentObjectsMinScale, environmentObjectsMaxScale)
                                                    .CompWiseMult(newEnvObj.transform.localScale);
                SetRandomColor(newEnvObj);
                environmentalObjects.Add(newEnvObj);
            }            
        }
    }
}
