using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShrinkAndDestroy : MonoBehaviour
{
    public float timeToStartShrink = 10;
    public float shrinkDuration = 0.5f;

    float timer;
    Vector3 origScale;

    // Start is called before the first frame update
    void Start()
    {
        origScale = transform.localScale;
    }

    // Update is called once per frame
    void Update()
    {
        timer += Time.deltaTime;
        if (timer >= timeToStartShrink)
        {
            transform.localScale = origScale * (1 - Mathf.Max(timer - timeToStartShrink, 0) / shrinkDuration);
        }
        if (timer >= shrinkDuration + timeToStartShrink)
        {
            Destroy(gameObject);
        }
    }
}
