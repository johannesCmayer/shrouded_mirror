using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StickForestGenerator : MonoBehaviour {

    public GameObject[] treePrefabs;
    public GameObject[] spawnBlockedArea;

    [Header("Config")]
    public int numberTrees = 10;

    private List<GameObject> spawnedObjects = new List<GameObject>();
    
    void Start()
    {
        SpawnForest(20);
    }

    public GameObject GetRandomGameObject(IList<GameObject> goList)
    {
        return goList[Random.Range(0, goList.Count)];
    }

    Vector3 RandPos(Transform areaTransform)
    {
        return areaTransform.position + new Vector3(
            Random.Range(-areaTransform.lossyScale.x / 2, areaTransform.lossyScale.x / 2),
            Random.Range(-areaTransform.lossyScale.y / 2, areaTransform.lossyScale.y / 2),
            Random.Range(-areaTransform.lossyScale.z / 2, areaTransform.lossyScale.z / 2));
    }

    public Vector3 GetRandomSpawnPos()
    {
        var randPos = RandPos(transform);
        return randPos;
    }

	public GameObject SpawnTree()
    {
        var newTree = Instantiate(GetRandomGameObject(treePrefabs), GetRandomSpawnPos(), Quaternion.identity);
        newTree.transform.Rotate(Random.Range(-0.1f, 0.1f), Random.Range(-0.1f, 0.1f), Random.Range(-0.1f, 0.1f),Space.World);
        newTree.transform.SetParent(gameObject.transform);
        return newTree;
    }

    public void SpawnForest(int numTrees)
    {
        for (int i = 0; i < numTrees; i++)
        {
            spawnedObjects.Add(SpawnTree());
        }
    }

    public void DeleteSpawnedObjects()
    {
        foreach (Transform item in transform)
        {
            if (!spawnedObjects.Contains(item.gameObject))
                spawnedObjects.Add(item.gameObject);
        }
        for (int i = spawnedObjects.Count - 1; i >= 0 ; i--)
        {
            DestroyImmediate(spawnedObjects[i]);
        }
    }
}
