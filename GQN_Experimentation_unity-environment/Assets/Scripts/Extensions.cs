using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Globalization;

public static class Extensions {

    public static string ToPreciseString(this Vector3 self)
    {
        return $"{self.x.ToString()}, {self.y.ToString()}, {self.z.ToString()}";
    }

    public static string ToPreciseString(this Quaternion self)
    {
        return $"{self.x.ToString()}, {self.y.ToString()}, {self.z.ToString()}, {self.w.ToString()}";
    }

    public static Vector3 CompWiseMult(this Vector3 me, Vector3 other)
    {
        return new Vector3(me.x * other.x, me.y * other.y, me.z * other.z);
    }

    public static Vector3 ArrayToVector3(this IList self, bool allowMoreThan3=false)
    {
        if (!allowMoreThan3 && self.Count > 3)
            throw new System.Exception("Array need 3 entries to be converted to vector 3, or explicitly allowed conversion");
        if (self is string[])
        {
            System.Func<object, float> f = x => float.Parse(x.ToString(), CultureInfo.InvariantCulture.NumberFormat);
            return new Vector3(f(self[0]), f(self[1]), f(self[2]));
        }
        else
        {
            return new Vector3((float)self[0], (float)self[1], (float)self[2]);
        }
    }
}
