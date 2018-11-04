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
    public int capturesBevoreRandomization = 100;
    public int maxNumEnvObjects = 1;
    public int minNumEnvObjects = 1;
    public float environmentObjectsMinScale = 0.4f;
    public float environmentObjectsMaxScale = 2f;
    [Header("Toogle randomization")]
    public bool randCamBackgroundColor = true;
    public bool randEnvColor = true;
    public bool randEnvObjColor = true;
    public bool randEnvObjPosition = true;
    public bool randEnvObjRotation = true;
    public bool randEnvObjScale = true;
    [Tooltip("If false always spawns objects equal to the max value.")]
    public bool randEnvObjectNumber = true;

    List<GameObject> environmentalObjects = new List<GameObject>();
    int captureCounter;
    int seed;
    string environmentID;

    public void UpdateEnvID()
    {
        //FieldInfo[] properties = typeof(EnvironmentGenerator).GetFields(
        //    BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);

        //long hash = ((float)seed).GetHashCode();
        //int i = 0;
        //foreach (var p in properties)
        //{
        //    if (!(p.FieldType == typeof(int) || 
        //        p.FieldType == typeof(bool) || 
        //        p.FieldType ==  typeof(float))) { continue; }

        //    float val;
        //    if (p.FieldType == typeof(bool))
        //    {
        //        val = (bool)p.GetValue(this) ? 1 : 0;
        //        if (val == 0)
        //            print("");
        //        else
        //            print("");
        //    }
        //    else
        //        val = (float)System.Convert.ChangeType(p.GetValue(this), typeof(float));
        //    hash += ((float)((i + 1) * 10 + val)).GetHashCode();
        //    hash = hash << 6;

        //    i++;
        //}
        environmentID = System.DateTime.UtcNow.ToString("yyyy-MM-dd-hh-mm-ss-fffffff-(zz)");
    }

    public string EnvironmentID {
        get {
            return environmentID;
        }
    }

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
        seed++;
        Random.InitState(seed);       
        if (randCamBackgroundColor)
            takeObservation.cam.backgroundColor = GetRandomColor();
        if (randEnvColor)
        {
            foreach (var item in environmentBaseObjects)
            {
                SetRandomColor(item);
            }
        }
        foreach (var item in environmentalObjectsPrefabs)
        {
            for (int i = environmentalObjects.Count - 1; i >= 0; i--)
            {
                Destroy(environmentalObjects[i]);
            }

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
        UpdateEnvID();
        EnvironmentChanged(EnvironmentID);
        print("Environment Randomized");
    }
}
