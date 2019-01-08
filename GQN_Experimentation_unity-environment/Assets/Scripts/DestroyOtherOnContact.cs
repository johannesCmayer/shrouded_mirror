using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DestroyOtherOnContact : MonoBehaviour
{
    public string[] tagsToDestroy = new[] { "Enemy" };

    private void OnTriggerEnter(Collider other)
    {        
        foreach (var tag in tagsToDestroy)
        {
            if (other.CompareTag(tag))
                other.GetComponent<IKillable>().Kill();
        }
    }
}
