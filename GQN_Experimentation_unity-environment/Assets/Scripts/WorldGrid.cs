using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WorldGrid : MonoBehaviour
{
    public GameObject gridBlockPrefab;
    public float gridBlockScale = 1;
    public int xSize = 30;
    public int zSize = 30;

    List<GameObject> gridBlocks = new List<GameObject>();

    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void DeleteGrid()
    {
        var toDestroy = new List<GameObject>();
        foreach (Transform item in transform)
        {
            if (item.GetComponent<WorldGrid>() == this)
                continue;
            toDestroy.Add(item.gameObject);
            if (gridBlocks.Contains(item.gameObject))
                gridBlocks.Remove(item.gameObject);
        }
        if (toDestroy.Count != 0)
        {
            for (int i = toDestroy.Count - 1; i <= 0; i--)
            {
                DestroyImmediate(toDestroy[i]);
            }
        }
    }

    public void CreateGrid()
    {
        //DeleteGrid();
        for (int i = 0; i < xSize; i++)
        {
            for (int j = 0; j < zSize; j++)
            {
                var spawnPos = transform.position + new Vector3(i * gridBlockScale, 0, j * gridBlockScale);
                var newObj = Instantiate(gridBlockPrefab, spawnPos, Quaternion.identity);
                newObj.transform.localScale = new Vector3(gridBlockScale, gridBlockScale, gridBlockScale);
                gridBlocks.Add(newObj);
                newObj.transform.SetParent(transform);
            }
        }
    }

    public void RaiseGridBlock(GameObject gridBlock)
    {
        if (gridBlock.transform.localPosition.y < 0.1f)
            gridBlock.transform.position += Vector3.up * gridBlockScale;
    }

    public void LowerGridBlock(GameObject gridBlock)
    {
        if (gridBlock.transform.localPosition.y > 0.9f)
            gridBlock.transform.position += Vector3.down * gridBlockScale;
    }
}
