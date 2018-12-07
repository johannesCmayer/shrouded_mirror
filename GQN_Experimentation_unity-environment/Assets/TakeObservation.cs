using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEngine.SceneManagement;
using UnityEditor;
using System.Linq;

[System.Serializable]
public class EnvironmentGroup
{
    public float weightToBeChosen = 1.0f;
    public List<Environment> environments = new List<Environment>();
}

public class TakeObservation : MonoBehaviour {

    public event System.Action TookObservation = delegate { };

    public Camera cam;
    public EnvironmentGenerator environmentGenerator;

    public List<EnvironmentGroup> environmentGroups;

    public int obsPerEnv = 32;
    public Vector3 offsetAfterPlacement;

    public string overwriteBasePath = $@"C:\trainingData";

    [Header("Camera Settings")]
    public bool rotateX = true;
    public bool rotateY = true;
    public bool rotateZ = false;
    
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
        var basePath = $@"{Application.dataPath}\..\..";
        if (overwriteBasePath != "")
            basePath = overwriteBasePath;
        return CreateDirectoryIfNotExists($@"{basePath}\trainingData\{sceneName}\" +
                                          $@"{cs.renderWidth}x{cs.renderHeight}\{environmentGenerator.EnvironmentID}");
    } 
    
    void Start()
    {
        myAS = GetComponent<AudioSource>();
        environmentGenerator.RandomizeEnv(cam);
        StartCoroutine(Capture());
	}

    public int GetRandomWeightedIndex(List<float> weights)
    {
        var combinedWeight = weights.Sum(item => item);
        var currentWeight = 0.0f;
        var floatIdx = Random.Range(0, combinedWeight);
        for (int i = 0; i < weights.Count; i++)
        {
            currentWeight += weights[i];
            if (currentWeight >= floatIdx)
                return i;
        }
        throw new System.Exception("No return value");
    }

    Transform GetRandomObserveAreaAndSetActiveations()
    {
        foreach (var envGroup in environmentGroups)
            foreach (var env in envGroup.environments)
                env.gameObject.SetActive(false);
        var observeAreas = new List<Transform>();
        for (int i = 0; observeAreas.Count == 0; i++)
        {
            if (i > 1000)
                throw new System.Exception("No env found with Observe Area");

            var environmentGroup = environmentGroups[GetRandomWeightedIndex(environmentGroups.Select(x => x.weightToBeChosen).ToList())];

            foreach (var env in environmentGroup.environments)
            {
                env.gameObject.SetActive(true);
                foreach (Transform child in env.transform)
                {
                    if (child.CompareTag("observeArea"))
                        observeAreas.Add(child);
                }
            }
        }
        var weights = observeAreas.Select(x => x.GetComponent<ObserveArea>().weightToBeChosen).ToList();
        return observeAreas[GetRandomWeightedIndex(weights)];
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

                    var capture = TakeObservationFromVolume(GetRandomObserveAreaAndSetActiveations(), cam);
                    SaveImage(capture, savePath, GetFileName(capture));
                    TookObservation();
                    totalImages++;
                    if (i % Mathf.Max(obsPerEnv, 1000) == 0)
                    {
                        print($"{(int)(((float)totalImages / totalImagesToMake) * 100)}% - " +
                            $"{(int)(totalImages / (Time.time - startTime))} images per second - " +
                            $"{totalImages}/{totalImagesToMake} are captured - " +
                            $"capturing now {cs.renderWidth}x{cs.renderHeight} images");                        
                    }
                    if (i % obsPerEnv == 0)
                    {
                        environmentGenerator.RandomizeEnv(cam);
                        yield return null;
                    }
                        
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
        camera.transform.position = new Util().GetRandomPointInAxisAlignedCube(transformVolume, offsetAfterPlacement);
        var randRot = new Vector3(
            rotateX ? Random.Range(0, 360) : camera.transform.rotation.eulerAngles.x,
            rotateY ? Random.Range(0, 360) : camera.transform.rotation.eulerAngles.y,
            rotateZ ? Random.Range(0, 360) : camera.transform.rotation.eulerAngles.z);
        camera.transform.rotation = Quaternion.Euler(randRot);
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