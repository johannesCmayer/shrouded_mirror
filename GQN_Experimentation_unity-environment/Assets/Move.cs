using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Jobs;

[ComputeJobOptimization]
public struct MovementJob : IJobParallelFor
{
    public float moveSpeed;
    public float deltaTime;

    public void Execute(int index)
    {

    }
}

public class Move : MonoBehaviour
{
    public Vector3 move = Vector3.forward;


    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        transform.position += (transform.forward * move.z + transform.right * move.x + transform.up * move.y) * Time.deltaTime;
    }
}
