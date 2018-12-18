Shader "Unlit/RedCutout"
{
	Properties
	{
		_MainTex("Albedo Texture", 2D) = "black" {}
		_RedCutoutThresh("Cutout Upper Threshold Red", Range(0.0,1.0)) = 1
		_CombinedCutoutThresh("Cutout Lower Threshold Combined", Range(0.0, 1.0)) = 0.01
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
		float _RedCutoutThresh;
		float _CombinendCutoutThresh;

		v2f vert(appdata v)
		{
			v2f o;
			o.vertex = UnityObjectToClipPos(v.vertex);
			o.uv = TRANSFORM_TEX(v.uv, _MainTex);
			return o;
		}

		fixed4 frag(v2f i) : SV_Target
		{
			float4 col = tex2D(_MainTex, i.uv);
			if (col.r >= _RedCutoutThresh || length(col.rgb) <= _CombinendCutoutThresh)
			{
				col.a = 0;
			}
			return col;
		}
			ENDCG
		}
		
}
}