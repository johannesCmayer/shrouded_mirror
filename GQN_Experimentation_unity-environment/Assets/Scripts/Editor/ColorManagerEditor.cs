using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CanEditMultipleObjects]
[CustomEditor(typeof(ColorManager))]
public class ColorManagerEditor : Editor
{
    Stack<Dictionary<GameObject, Material>> undos = new Stack<Dictionary<GameObject, Material>>();

    public override void OnInspectorGUI()
    {
        var myTarget = (ColorManager)target;
        myTarget.SetSingelton();
        if (GUILayout.Button("Set Random Material On Selected Objects"))
        {
            var dict = new Dictionary<GameObject, Material>();
            foreach (var go in Selection.gameObjects)
            {
                var rend = go.GetComponent<Renderer>();
                if (rend != null)
                {
                    dict.Add(go, rend.sharedMaterial);
                    ColorManager.instance.SetRandomPoolMaterialOnGO(go);
                }
            }
            undos.Push(dict);
        }
        if (GUILayout.Button("Undo"))
        {
            if (undos.Count == 0)
            {
                Debug.Log("No More Undos");
            }
            else
            {
                var undo = undos.Pop();
                foreach (var go in undo.Keys)
                {
                    go.GetComponent<Renderer>().sharedMaterial = undo[go];
                }
            }
        }
        if (GUILayout.Button("Set Materials to Color Pool Colors"))
        {
            myTarget.SetColorsOnMaterialPoolToPoolColors();
        }

        DrawDefaultInspector();
    }
}
