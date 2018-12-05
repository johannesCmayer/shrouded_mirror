using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Reflection;

public class EnvironmentGenerator : MonoBehaviour {

    public event System.Action<string> EnvironmentChanged = delegate { };

    [Header("Setup")]
    public TakeObservation takeObservation;
    public Transform environmentalObjectsSpawnVolume;
    public GameObject[] environmentBaseObjects;
    public GameObject[] environmentalObjectsPrefabs;
    [Header("Settings")]
    public int maxNumEnvObjects = 1;
    public int minNumEnvObjects = 1;
    public float environmentObjectsMinScale = 0.4f;
    public float environmentObjectsMaxScale = 2f;
    public bool useColorPool = true;
    public Color[] colorPool;
    [Header("Basic color randomisation")]
    public bool randCamBackgroundColor = true;
    public bool randEnvColor = true;
    public bool randEnvMatTilling = true;
    public Vector2 minMaxTiling = new Vector3(2, 6);
    public bool allWallsSameColor = true;
    [Header("Object randomisation")]
    public bool spawnEnvObjects = false;
    public bool randEnvObjColor = true;
    public bool randEnvObjPosition = true;
    public bool randEnvObjRotation = true;
    public bool randEnvObjScale = true;
    [Tooltip("If false always spawns objects equal to the max value.")]
    public bool randEnvObjectNumber = true;

    List<GameObject> environmentalObjects = new List<GameObject>();
    int captureCounter;
    string environmentID;

    public string EnvironmentID
    {
        get
        {
            return environmentID;
        }
    }

    public void RandomizeEnv(Camera observationCamera)
    {
        if (randCamBackgroundColor)
            observationCamera.backgroundColor = GetRandomColor();
        if (randEnvColor)
        {
            if (allWallsSameColor)
            {
                var color = GetRandomColor();
                foreach (var item in environmentBaseObjects)
                {
                    item.GetComponent<Renderer>().material.color = color;
                }
            }
            else
            {
                foreach (var item in environmentBaseObjects)
                {
                    SetRandomColor(item);
                }
            }
        }
        if (randEnvMatTilling)
        {
            var scale = Random.Range((int)minMaxTiling.x, (int)minMaxTiling.y + 1);
            foreach (var item in environmentBaseObjects)
            {                
                item.GetComponent<Renderer>().material.mainTextureScale = new Vector2(scale, scale);
            }
        }
        for (int i = environmentalObjects.Count - 1; i >= 0; i--)
        {
            Destroy(environmentalObjects[i]);
        }
        if (spawnEnvObjects)
        {
            foreach (var item in environmentalObjectsPrefabs)
            {
                int numObjs;
                if (randEnvObjectNumber)
                    numObjs = Random.Range(minNumEnvObjects, maxNumEnvObjects + 1);
                else
                    numObjs = maxNumEnvObjects;
                for (int i = 0; i < numObjs; i++)
                {
                    var idx = Random.Range(0, environmentalObjectsPrefabs.Length);
                    var newEnvObj = Instantiate(environmentalObjectsPrefabs[idx], Vector3.zero, Quaternion.identity);
                    if (randEnvObjPosition)
                        newEnvObj.transform.position = new Util().GetRandomPointInAxisAlignedCube(environmentalObjectsSpawnVolume);
                    if (randEnvObjRotation)
                        newEnvObj.transform.rotation = Random.rotation;
                    if (randEnvObjScale)
                        newEnvObj.transform.localScale = GetRandomVec3(environmentObjectsMinScale, environmentObjectsMaxScale)
                                                        .CompWiseMult(newEnvObj.transform.localScale);
                    if (randEnvObjColor)
                        SetRandomColor(newEnvObj);
                    environmentalObjects.Add(newEnvObj);
                }
            }
        }
        UpdateEnvID();
        EnvironmentChanged(EnvironmentID);
    }

    public void UpdateEnvID()
    {
        environmentID = System.DateTime.UtcNow.ToString("yyyy-MM-dd-hh-mm-ss-fffffff-(zz)");
    }    

    private Vector3 GetRandomVec3(float rangeMin=0, float rangeMax=1)
    {
        System.Func<float> r_val = () => Random.Range(rangeMin, rangeMax);
        return new Vector3(r_val(), r_val(), r_val());
    }

    private Color GetRandomColor(float rangeMin = 0, float rangeMax = 1)
    {
        if (useColorPool)
        {
            return colorPool[Random.Range(0, colorPool.Length)];
        }
        else
        {
            System.Func<float> r_val = () => Random.Range(rangeMin, rangeMax);
            return new Color(r_val(), r_val(), r_val());
        }
    }

    private void SetRandomColor(GameObject go)
    {
        go.GetComponent<Renderer>().material.color = GetRandomColor();
    }

    
}
