Shader "Custom/CropShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Crop ("Crop", Vector) = (0, 0, 1, 1)
    }

    SubShader
    {
        Pass
        {

            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _Crop;

            fixed4 frag(v2f_img i) : SV_Target
            {
                float2 uv = i.uv * _Crop.zw + _Crop.xy;
                return tex2D(_MainTex, uv);
            }
            ENDCG
        }
    }
}
