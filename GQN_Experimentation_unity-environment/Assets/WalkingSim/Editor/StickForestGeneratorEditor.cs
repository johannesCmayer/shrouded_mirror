using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(StickForestGenerator))]
[CanEditMultipleObjects]
public class StickForestGeneratorEditor : Editor {

    public override void OnInspectorGUI()
    {
        StickForestGenerator myTarget = (StickForestGenerator)target;
        if (GUILayout.Button("Genrate New"))
        {
            myTarget.DeleteSpawnedObjects();
            myTarget.SpawnForest(myTarget.numberTrees);
        }
        if (GUILayout.Button("deleteall"))
        {
            myTarget.DeleteSpawnedObjects();
        }
        DrawDefaultInspector();
    }
}
