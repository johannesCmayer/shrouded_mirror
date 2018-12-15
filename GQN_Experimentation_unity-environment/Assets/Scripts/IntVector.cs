using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public struct IntVector
{
    int x, y, z;

    public static IntVector forward = new IntVector(0, 0, 1);
    public static IntVector right = new IntVector(1, 0, 0);
    public static IntVector back = new IntVector(0, 0, -1);
    public static IntVector left = new IntVector(-1, 0, 0);
    public static IntVector up = new IntVector(0, 1, 0);
    public static IntVector down = new IntVector(0, -1, 0);

    public IntVector(int x, int y, int z)
    {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public override bool Equals(object obj)
    {
        if (obj is IntVector)
        {
            var vec = (IntVector)obj;
            return
                vec.x == x &&
                vec.y == y &&
                vec.z == z;
        }
        return false;
    }

    public int Pow(int x, int p)
    {
        for (int i = 0; i < p; i++)
            x *= x;
        return (x);
    }

    public override int GetHashCode()
    {
        return Pow(2, x) * Pow(3, y) * Pow(5, z);
    }

    public Vector3 ToVector3()
    {
        return new Vector3(x, y, z);
    }

    public static IntVector operator +(IntVector a, IntVector b)
    {
        return new IntVector(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    public static bool operator ==(IntVector a, IntVector b)
    {
        return a.Equals(b);
    }

    public static bool operator !=(IntVector a, IntVector b)
    {
        return a.Equals(b);
    }
}
