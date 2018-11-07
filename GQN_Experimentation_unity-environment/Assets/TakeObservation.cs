using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEngine.SceneManagement;
using UnityEditor;

public class TakeObservation : MonoBehaviour {

    public event System.Action TookObservation = delegate { };

    public Camera cam;
    public Transform observeVolume;
    public EnvironmentGenerator environmentGenerator;

    public int captureBatchSize = 32;

    public CaptureSettings[] captureSettings = new CaptureSettings[] {
        new CaptureSettings(8,8,20000),
        new CaptureSettings(16,16,20000),
        new CaptureSettings(32,32,20000),
        new CaptureSettings(64,64,20000),
        new CaptureSettings(128,128,20000),
        new CaptureSettings(256,256,20000)
    };

    AudioSource myAS;

    private string DefaultSavePath(string sceneName, CaptureSettings cs)
    {
        return CreateDirectoryIfNotExists($@"{Application.dataPath}\..\..\trainingData\{sceneName}\" +
                                          $@"{cs.renderWidth}x{cs.renderHeight}\{environmentGenerator.EnvironmentID}");
    } 
    
    void Start()
    {
        myAS = GetComponent<AudioSource>();
        StartCoroutine(Capture());
	}

    IEnumerator Capture()
    {
        for (int i = 3; i >= 0; i--)
        {
            yield return new WaitForSeconds(1);
            print($"capture starts in {i}s");
            myAS.Play();            
        }
        int totalImages = 0;
        int totalImagesToMake = 0;
        foreach (var cs in captureSettings)
        {
            if (cs.execute)
                totalImagesToMake += cs.numImagesToMake;
        }
        var startTime = Time.time;
        foreach (var cs in captureSettings)
        {
            if (cs.execute)
            {                
                cam.targetTexture = new RenderTexture(cs.renderWidth, cs.renderHeight, 1);

                for (int i = 0; i < cs.numImagesToMake; i++)
                {
                    var sceneName = SceneManager.GetActiveScene().name;
                    var savePath = DefaultSavePath(sceneName, cs);
                    var capture = TakeObservationFromVolume(observeVolume, cam);
                    SaveImage(capture, savePath, GetFileName(capture));
                    TookObservation();
                    totalImages++;
                    if (i % Mathf.Max(captureBatchSize, 1000) == 0)
                    {
                        print($"{(int)(((float)totalImages / totalImagesToMake) * 100)}% - " +
                            $"{(int)(totalImages / (Time.time - startTime))} images per second - " +
                            $"{totalImages}/{totalImagesToMake} are captured - " +
                            $"capturing now {cs.renderWidth}x{cs.renderHeight} images");                        
                    }
                    if (i % captureBatchSize == 0)
                        yield return null;
                }
            }
        }
        print($"Capture completed, {totalImages} images generated in {Time.time - startTime}s");
        for (int i = 0; i < 8; i++)        
        {
            myAS.Play();            
            yield return new WaitForSeconds(0.2f);
        }
        UnityEditor.EditorApplication.isPlaying = false;
    }

    public string CreateDirectoryIfNotExists(string path)
    {
        if (!Directory.Exists(path))
            Directory.CreateDirectory(path);
        return path;
    }

    public CaptureData TakeObservationFromVolume(Transform transformVolume, Camera camera)
    {
        camera.transform.position = new Util().GetRandomPointInAxisAlignedCube(transformVolume);
        camera.transform.rotation = Quaternion.Euler(0, Random.Range(0, 360), 0);
        return AdvancedCameraObservation(camera);
    }    

    public string GetFileName(CaptureData obs)
    {
        var currentUTCTime = System.DateTime.UtcNow.ToString("yyyy-MM-dd-hh-mm-ss-fffffff-(zz)");
        return $@"{ obs.position.ToPreciseString()}_{ obs.rotation.ToPreciseString()}_{ currentUTCTime}.png";
    }

    public void SaveImage(CaptureData obs, string saveDir, string fileName)
    {
        File.WriteAllBytes($@"{saveDir}\{fileName}", obs.png);
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

    public CaptureSettings(bool execute, int renderWidth, int renderHeight, int numImagesToMake)
    {
        this.execute = execute;
        this.renderWidth = renderWidth;
        this.renderHeight = renderHeight;
        this.numImagesToMake = numImagesToMake;
    }
}