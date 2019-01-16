using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GenerateObserveAreas : MonoBehaviour
{
    public GameObject unitObserveAreaPrefab;

    public void Generate()
    {
        var parent = new GameObject("ENV_Grid_Generated", new[] { typeof(Environment) });
        var children = new List<GameObject>();
        foreach (Transform trans in transform)
        {
            children.Add(trans.gameObject);
        }
        foreach (var child in children)
        {
            if (child.transform.localPosition.y < 0.1)
            {
                var area = Instantiate(unitObserveAreaPrefab, new Vector3 (child.transform.position.x, 1.5f, child.transform.position.z), Quaternion.identity);
                area.transform.SetParent(parent.transform);
                area.transform.localScale = new Vector3(child.transform.localScale.x, 0.01f, child.transform.localScale.z);
            }
        }
    }
}
