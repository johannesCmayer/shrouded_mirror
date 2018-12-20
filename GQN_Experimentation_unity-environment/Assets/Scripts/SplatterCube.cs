using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SplatterCube : MonoBehaviour
{
    public GameObject Fragment;
    public int thirdRootOfNumberOfFragments = 3;

    void Start()
    {
        
    }
    
    void Update()
    {
        
    }

    void SpawnSplatter()
    {
        for (int axis = 0; axis < 3; axis++)
        {
            for (int j = 0; j < thirdRootOfNumberOfFragments; j++)
            {
                var pos = transform.position[axis] + transform.localScale[axis] * Mathf.Lerp(-1, 1, j / thirdRootOfNumberOfFragments);
            }
        }
    }

    private void OnDestroy()
    {
        
    }
}
