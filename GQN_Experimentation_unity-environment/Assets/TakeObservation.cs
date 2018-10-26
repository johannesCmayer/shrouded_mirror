using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEngine.SceneManagement;

public class TakeObservation : MonoBehaviour {

    public Camera camera;
    public Transform observeVolume;
    public int renderWidth = 100;
    public int renderHeight = 100;
    public string overrideSavePath;

    private string DefaultSavePath(string sceneName) { return $@"{Application.dataPath}\..\..\trainingData\{sceneName}_{renderWidth}x{renderHeight}"; } 

    int imageCounter;
    float currentTime;

    void Start()
    {
        if (camera.targetTexture == null)
        {
            camera.targetTexture = new RenderTexture(renderWidth, renderHeight, 1);
        }
        print(Application.dataPath);
        
	}
	
	void Update ()
    {        
        var sceneName = SceneManager.GetActiveScene().name;
        string savePath = "";
        savePath = CreateDirectoryIfNotExists(savePath != string.Empty ? savePath : DefaultSavePath(sceneName));
        SaveImage(TakeObservationFromVolume(observeVolume, camera), savePath);
        imageCounter++;

        currentTime = Time.time;
    }

    private void OnDestroy()
    {
        print($"{imageCounter} images produced in {currentTime}s.");
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
