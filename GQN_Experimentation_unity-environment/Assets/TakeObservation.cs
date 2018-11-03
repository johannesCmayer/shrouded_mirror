using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEngine.SceneManagement;

[System.Serializable]
public class CaptureSettings
{
    public bool execute = true;
    public int renderWidth = 128;
    public int renderHeight = 128;
    public int numImagesToMake = 20000;

    public CaptureSettings()
    {
    }

    public CaptureSettings(int renderWidth, int renderHeight, int number_of_img)
    {
        this.renderWidth = renderWidth;
        this.renderHeight = renderHeight;
        this.numImagesToMake = number_of_img;
    }
}

public class TakeObservation : MonoBehaviour {

    public Camera camera;
    public Transform observeVolume;

    public int captureBatchSize = 32;
    public int statusPrintFreqCoef = 1;

    public CaptureSettings[] captureSettings = new CaptureSettings[] {
        new CaptureSettings(32,32,20000),
        new CaptureSettings(64,64,20000),
        new CaptureSettings(128,128,20000),
        new CaptureSettings(256,256,20000)
    };

    private string DefaultSavePath(string sceneName, CaptureSettings cs)
    {
        return $@"{Application.dataPath}\..\..\trainingData\{sceneName}_{cs.renderWidth}x{cs.renderHeight}";
    } 
    
    void Start()
    {
        StartCoroutine(Capture());
	}

    IEnumerator Capture()
    {        
        int totalImages = 0;
        int totalImagesToMake = 0;
        foreach (var cs in captureSettings)
        {
            totalImagesToMake += cs.numImagesToMake;
        }
        var startTime = Time.time;
        foreach (var cs in captureSettings)
        {
            if (cs.execute)
            {                
                camera.targetTexture = new RenderTexture(cs.renderWidth, cs.renderHeight, 1);

                for (int i = 0; i < cs.numImagesToMake; i++)
                {
                    var sceneName = SceneManager.GetActiveScene().name;
                    var savePath = CreateDirectoryIfNotExists(DefaultSavePath(sceneName, cs));
                    SaveImage(TakeObservationFromVolume(observeVolume, camera), savePath);
                    totalImages++;
                    if (i % captureBatchSize == 0)
                    {
                        print($"{(int)(((float)totalImages / totalImagesToMake) * 100)}% - " +
                            $"{(int)(totalImages / (Time.time - startTime))} images per second - " +
                            $"{totalImages}/{totalImagesToMake} are captured");
                        yield return null;
                    }
                }
            }
        }
    }

    public string CreateDirectoryIfNotExists(string path)
    {
        if (!Directory.Exists(path))
            Directory.CreateDirectory(path);
        return path;
    }

    public CaptureData TakeObservationFromVolume(Transform transformVolume, Camera camera)
    {
        camera.transform.position = GetRandomPointInAxisAlignedCube(transformVolume);
        camera.transform.rotation = Quaternion.Euler(0, Random.Range(0, 360), 0);
        return AdvancedCameraObservation(camera);
    }

    public Vector3 GetRandomPointInAxisAlignedCube(Transform cubeTransform)
    {
        var cs = cubeTransform.localScale;
        return cubeTransform.transform.position + new Vector3(
            Random.Range(-cs.x / 2, cs.x / 2), 
            Random.Range(-cs.y / 2, cs.y / 2), 
            Random.Range(-cs.z / 2, cs.z / 2));        
    }

    public void SaveImage(CaptureData obs, string saveDir)
    {
        var currentUTCTime = System.DateTime.UtcNow.ToString("yyyy-MM-dd-hh-mm-ss-fffffff-(zz)");
        File.WriteAllBytes($@"{saveDir}\{obs.position.ToPreciseString()}_{obs.rotation.ToPreciseString()}_{currentUTCTime}.png", obs.png);
    }

    public CaptureData AdvancedCameraObservation(Camera camera)
    {
        return new CaptureData(TakeCameraObservation(camera), camera.transform.position, camera.transform.rotation);
    }

    public Texture2D TakeCameraObservation(Camera camera)
    {
        RenderTexture currentRenderTexture = RenderTexture.active;
        RenderTexture.active = camera.targetTexture;
        camera.Render();
        Texture2D image = new Texture2D(camera.targetTexture.width, camera.targetTexture.height);
        image.ReadPixels(new Rect(0, 0, camera.targetTexture.width, camera.targetTexture.height),0 , 0);
        image.Apply();
        RenderTexture.active = currentRenderTexture;
        return image;        
    }
}

public struct CaptureData
{
    public readonly Texture2D texture;
    public byte[] png
    {
        get { return texture.EncodeToPNG(); }
    }
    public readonly Vector3 position;
    public readonly Quaternion rotation;

    public CaptureData(Texture2D texture, Vector3 position, Quaternion rotation)
    {
        this.texture = texture;
        this.position = position;
        this.rotation = rotation;
    }
}
