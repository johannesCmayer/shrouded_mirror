//Shader "Transparent/Cutout/Transparent" {
//	Properties{
//		_Color("Main Color", Color) = (1,1,1,1)
//		_MainTex("Base (RGB) Trans (A)", 2D) = "white" {}
//	_CutTex("Cutout (A)", 2D) = "white" {}
//	_Cutoff("Red cutoff", Range(0, 1)) = 1
//	}
//
//SubShader{
//	Tags{ "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent" }
//	LOD 200
//
//	CGPROGRAM
//#pragma surface surf Lambert alpha
//
//	sampler2D _MainTex;
//	sampler2D _CutTex;
//	fixed4 _Color;
//	float _Cutoff;
//
//	struct Input {
//		float2 uv_MainTex;
//	};
//
//	void surf(Input IN, inout SurfaceOutput o) {
//		fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * _Color;		
//		float cutR = tex2D(_CutTex, IN.uv_MainTex).r;
//		o.Albedo = c.rgb;
//
//		if (_Color.r >= _Cutoff)
//		{
//			o.Alpha = 0;
//		}
//		else
//		{
//
//			o.Alpha = c.a;
//		}
//	}
//
//	ENDCG
//}
//
//		Fallback "Transparent/VertexLit"
//}

Shader "Unlit/SpecialFX/Cool Hologram"
{
	Properties
	{
		_MainTex("Albedo Texture", 2D) = "black" {}
	_TintColor("Tint Color", Color) = (0,1,1,1)
		_CutoutThresh("Cutout Threshold", Range(0.0,1.0)) = 0.2
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
	float4 _TintColor;
	float _CutoutThresh;

	v2f vert(appdata v)
	{
		v2f o;
		o.vertex = UnityObjectToClipPos(v.vertex);
		o.uv = TRANSFORM_TEX(v.uv, _MainTex);
		return o;
	}

	fixed4 frag(v2f i) : SV_Target
	{
		fixed4 col = tex2D(_MainTex, i.uv);
		if (col.r >= _CutoutThresh)
		{
			col.a = 0;
		}
		else
		{
			col.a = 1;
		}
		return col;
	}
		ENDCG
	}
	}
}