using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Simple3DGizmo : MonoBehaviour
{
    public bool draw = true;

    private void OnDrawGizmos()
    {
        if (draw)
        {
            Gizmos.color = Color.red * 0.1f;
            Gizmos.DrawCube(transform.position, transform.localScale * 1.1f);
            Gizmos.DrawWireCube(transform.position, transform.localScale * 1f);
        }
    }
}
