using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class UndoStep
{
    public Dictionary<GameObject, Material> colorDict = new Dictionary<GameObject, Material>();
}

public class ColorManager : MonoBehaviour {
    public static ColorManager instance;
    
    public List<Material> materialPool = new List<Material>();
    public List<Color> colorPool = new List<Color>()
    {
        new Color(1,0,0),
        new Color(0,1,0),
        new Color(1,0,1),
        new Color(1,1,0),
    };

    public void SetSingelton()
    {
        instance = this;
    }

    void Start () {
        SetSingelton();
	}
	
    public void SetColorsOnMaterialPoolToPoolColors()
    {
        for (int i = 0; i < materialPool.Count; i++)
        {
            materialPool[i].color = colorPool[i];
        }
    }

	public Material GetRandomMaterialFromPool()
    {
        return materialPool[Random.Range(0, materialPool.Count)];
    }

    public void SetRandomPoolMaterialOnGO(GameObject obj)
    {
        obj.GetComponent<Renderer>().sharedMaterial = GetRandomMaterialFromPool();
    }
}
