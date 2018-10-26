using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class Extensions {

    public static string ToPreciseString(this Vector3 self)
    {
        return $"{self.x.ToString()}, {self.y.ToString()}, {self.z.ToString()}";
    }

    public static string ToPreciseString(this Quaternion self)
    {
        return $"{self.x.ToString()}, {self.y.ToString()}, {self.z.ToString()}, {self.w.ToString()}";
    }
}
