using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading;
using System.Globalization;

public class CultureSetter : MonoBehaviour {

    private void Awake()
    {
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
    }
}
