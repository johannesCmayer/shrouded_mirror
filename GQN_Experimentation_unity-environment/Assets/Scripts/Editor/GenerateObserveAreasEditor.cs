using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(GenerateObserveAreas))]
public class GenerateObserveAreasEditor : Editor
{
    public override void OnInspectorGUI()
    {
        var myTarget = (GenerateObserveAreas)target;
        if (GUILayout.Button("Generate"))
            myTarget.Generate();

        DrawDefaultInspector();
    }
}
