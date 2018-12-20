Shader "Unlit/KeyOut"
{
	Properties
	{
		_MainTex("Albedo Texture", 2D) = "black" {}
		_CC1("Keyout color 1", Color) = (1,0,0,1)
		_CC2("Keyout color 2", Color) = (0,0,0,1)
		_NTC("Not Transparent color", Color) = (1, 0, 0, 1)
		_Trans("Transparency of output", Float) = 0.5
		_Tol("Tolerance", Float) = 0.01
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
		float4 _CC1;
		float4 _CC2;
		float1 _Trans;
		float _Tol;
		float4 _NTC;

		v2f vert(appdata v)
		{
			v2f o;
			o.vertex = UnityObjectToClipPos(v.vertex);
			o.uv = TRANSFORM_TEX(v.uv, _MainTex);
			return o;
		}

		float cull(float3 col, float3 culCol)
		{
			if (col.r <= culCol.r + _Tol && col.r >= culCol.r - _Tol &&
				col.g <= culCol.g + _Tol && col.g >= culCol.g - _Tol &&
				col.b <= culCol.b + _Tol && col.b >= culCol.b - _Tol)
			{
				return 1;
			}
			else
			{
				return 0;
			}
		}

		fixed4 frag(v2f i) : SV_Target
		{
			float4 col = tex2D(_MainTex, i.uv);
			if (cull(col.rgb, _CC1.rgb) + cull(col.rgb, _CC2.rgb) > 0)
			{
				col.a = 0;
			}
			else if (
				abs(col.r - _NTC.r) < _Tol,
				abs(col.g - _NTC.g) < _Tol,
				abs(col.b - _NTC.b) < _Tol)
			{
				col.a = 1;
			}
			else
			{
				col.a = _Trans;
			}
			return col;
		}
			ENDCG
		}
		
}
}