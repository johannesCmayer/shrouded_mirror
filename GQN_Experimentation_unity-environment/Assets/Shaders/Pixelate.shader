Shader "Effect/Pixelate"
{
	Properties
	{
		_MainTex("Albedo Texture", 2D) = "black" {}
		_PixelResX("Pixelation X", Int) = 32
		_PixelResY("Pixelation Y", Int) = 32
	}

		SubShader
	{
		Tags{ "Queue" = "Transparent" "RenderType" = "Transparent" }
		LOD 100

		ZTest Always Cull Off ZWrite Off
		Fog{ Mode off }
		Blend SrcAlpha OneMinusSrcAlpha

		Pass
	{
		CGPROGRAM
#pragma vertex vert
#pragma fragment frag

#include "UnityCG.cginc"

		struct appdata
	{
		float4 vertex : POSITION;
		float2 uv : TEXCOORD0;
	};

	struct v2f
	{
		float2 uv : TEXCOORD0;
		float4 vertex : SV_POSITION;
	};

	sampler2D _MainTex;
	float4 _MainTex_ST;
	float _PixelResX;
	float _PixelResY;

	v2f vert(appdata v)
	{
		v2f o;
		o.vertex = UnityObjectToClipPos(v.vertex);
		o.uv = TRANSFORM_TEX(v.uv, _MainTex);
		return o;
	}

	float2 pixlate(float2 cord)
	{
		return float2(floor(cord.x * _PixelResX) / _PixelResX,
			floor(cord.y * _PixelResY) / _PixelResY);
	}

	fixed4 frag(v2f i) : SV_Target
	{
		float4 col = tex2D(_MainTex, pixlate(i.uv));
		return col;
	}
		ENDCG
	}

	}
}