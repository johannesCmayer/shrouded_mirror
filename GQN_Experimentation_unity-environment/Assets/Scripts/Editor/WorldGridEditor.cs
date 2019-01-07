using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(WorldGrid))]
public class WorldGridEditor : Editor
{
    public override void OnInspectorGUI()
    {
        var myTarget = (WorldGrid)target;
        if (GUILayout.Button("NewGrid"))
            myTarget.CreateGrid();
        DrawDefaultInspector();
    }
}
