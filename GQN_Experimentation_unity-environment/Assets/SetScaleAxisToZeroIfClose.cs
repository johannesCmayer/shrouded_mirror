using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SetScaleAxisToZeroIfClose : MonoBehaviour {

    public float setToZeroMaxSize = 0.11f;

	void Start () {
        var newScaleVec = transform.localScale;

        if (transform.localScale.x < setToZeroMaxSize)
            newScaleVec.x = 0;

        if (transform.localScale.y < setToZeroMaxSize)
            newScaleVec.y = 0;

        if (transform.localScale.z < setToZeroMaxSize)
            newScaleVec.z = 0;

        transform.localScale = newScaleVec;
    }
}
