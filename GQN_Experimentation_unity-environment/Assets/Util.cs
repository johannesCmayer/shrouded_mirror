using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Util {

    public Vector3 GetRandomPointInAxisAlignedCube(Transform cubeTransform)
    {
        var cs = cubeTransform.localScale;
        return cubeTransform.transform.position + new Vector3(
            Random.Range(-cs.x / 2, cs.x / 2),
            Random.Range(-cs.y / 2, cs.y / 2),
            Random.Range(-cs.z / 2, cs.z / 2));
    }
}
