//
//  Shaders.metal
//  LokLokARMetal
//
//  Created by LOK on 3/8/2017.
//  Copyright © 2017 WONG LOK. All rights reserved.
//

#include <metal_stdlib>
#include <simd/simd.h>

// Include header shared between this Metal shader code and C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

typedef struct {
    float2 position [[attribute(kVertexAttributePosition)]];
    float2 texCoord [[attribute(kVertexAttributeTexcoord)]];
} ImageVertex;


typedef struct {
    float4 position [[position]];
    float2 texCoord;
} ImageColorInOut;


// Captured image vertex function
vertex ImageColorInOut capturedImageVertexTransform(ImageVertex in [[stage_in]]) {
    ImageColorInOut out;
    
    // Pass through the image vertex's position
    out.position = float4(in.position, 0.0, 1.0);
    
    // Pass through the texture coordinate
    out.texCoord = in.texCoord;
    
    return out;
}

// Captured image fragment function
fragment float4 capturedImageFragmentShader(ImageColorInOut in [[stage_in]],
                                            texture2d<float, access::sample> capturedImageTextureY [[ texture(kTextureIndexY) ]],
                                            texture2d<float, access::sample> capturedImageTextureCbCr [[ texture(kTextureIndexCbCr) ]]) {
    
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);
    
    const float4x4 ycbcrToRGBTransform = float4x4(
        float4(+1.0000f, +1.0000f, +1.0000f, +0.0000f),
        float4(+0.0000f, -0.3441f, +1.7720f, +0.0000f),
        float4(+1.4020f, -0.7141f, +0.0000f, +0.0000f),
        float4(-0.7010f, +0.5291f, -0.8860f, +1.0000f)
    );
    
    // Sample Y and CbCr textures to get the YCbCr color at the given texture coordinate
    float4 ycbcr = float4(capturedImageTextureY.sample(colorSampler, in.texCoord).r,
                          capturedImageTextureCbCr.sample(colorSampler, in.texCoord).rg, 1.0);
    
    // Return converted RGB color
    return ycbcrToRGBTransform * ycbcr;
}


typedef struct {
    float3 position [[attribute(kVertexAttributePosition)]];
    float2 texCoord [[attribute(kVertexAttributeTexcoord)]];
    half3 normal    [[attribute(kVertexAttributeNormal)]];
} Vertex;


typedef struct {
    float4 position [[position]];
    float4 color;
    half3  eyePosition;
    half3  normal;
} ColorInOut;


// Anchor geometry vertex function
vertex ColorInOut anchorGeometryVertexTransform(Vertex in [[stage_in]],
                                                constant SharedUniforms &sharedUniforms [[ buffer(kBufferIndexSharedUniforms) ]],
                                                constant InstanceUniforms *instanceUniforms [[ buffer(kBufferIndexInstanceUniforms) ]],
                                                ushort vid [[vertex_id]],
                                                ushort iid [[instance_id]]) {
    ColorInOut out;
    
    // Make position a float4 to perform 4x4 matrix math on it
    float4 position = float4(in.position, 1.0);
    
    float4x4 modelMatrix = instanceUniforms[iid].modelMatrix;
    float4x4 modelViewMatrix = sharedUniforms.viewMatrix * modelMatrix;
    
    // Calculate the position of our vertex in clip space and output for clipping and rasterization
    out.position = sharedUniforms.projectionMatrix * modelViewMatrix * position;
    
    // Color each face a different color
    ushort colorID = vid / 4 % 6;
    out.color = colorID == 0 ? float4(0.0, 1.0, 0.0, 1.0) // Right face
              : colorID == 1 ? float4(1.0, 0.0, 0.0, 1.0) // Left face
              : colorID == 2 ? float4(0.0, 0.0, 1.0, 1.0) // Top face
              : colorID == 3 ? float4(1.0, 0.5, 0.0, 1.0) // Bottom face
              : colorID == 4 ? float4(1.0, 1.0, 0.0, 1.0) // Back face
              : float4(1.0, 1.0, 1.0, 1.0); // Front face
    
    // Calculate the positon of our vertex in eye space
    out.eyePosition = half3((modelViewMatrix * position).xyz);
    
    // Rotate our normals to world coordinates
    float4 normal = modelMatrix * float4(in.normal.x, in.normal.y, in.normal.z, 0.0f);
    out.normal = normalize(half3(normal.xyz));
    
    return out;
}

// Anchor geometry fragment function
fragment float4 anchorGeometryFragmentLighting(ColorInOut in [[stage_in]],
                                               constant SharedUniforms &uniforms [[ buffer(kBufferIndexSharedUniforms) ]]) {
    
    float3 normal = float3(in.normal);
    
    // Calculate the contribution of the directional light as a sum of diffuse and specular terms
    float3 directionalContribution = float3(0);
    {
        // Light falls off based on how closely aligned the surface normal is to the light direction
        float nDotL = saturate(dot(normal, -uniforms.directionalLightDirection));
        
        // The diffuse term is then the product of the light color, the surface material
        // reflectance, and the falloff
        float3 diffuseTerm = uniforms.directionalLightColor * nDotL;
        
        // Apply specular lighting...
        
        // 1) Calculate the halfway vector between the light direction and the direction they eye is looking
        float3 halfwayVector = normalize(-uniforms.directionalLightDirection - float3(in.eyePosition));
        
        // 2) Calculate the reflection angle between our reflection vector and the eye's direction
        float reflectionAngle = saturate(dot(normal, halfwayVector));
        
        // 3) Calculate the specular intensity by multiplying our reflection angle with our object's
        //    shininess
        float specularIntensity = saturate(powr(reflectionAngle, uniforms.materialShininess));
        
        // 4) Obtain the specular term by multiplying the intensity by our light's color
        float3 specularTerm = uniforms.directionalLightColor * specularIntensity;
        
        // Calculate total contribution from this light is the sum of the diffuse and specular values
        directionalContribution = diffuseTerm + specularTerm;
    }
    
    // The ambient contribution, which is an approximation for global, indirect lighting, is
    // the product of the ambient light intensity multiplied by the material's reflectance
    float3 ambientContribution = uniforms.ambientLightColor;
    
    // Now that we have the contributions our light sources in the scene, we sum them together
    // to get the fragment's lighting value
    float3 lightContributions = ambientContribution + directionalContribution;
    
    // We compute the final color by multiplying the sample from our color maps by the fragment's
    // lighting value
    float3 color = in.color.rgb * lightContributions;
    
    // We use the color we just computed and the alpha channel of our
    // colorMap for this fragment's alpha value
    return float4(color, in.color.w);
}

float constrain(float val, float min, float max) {
    if (val < min) {
        return min;
    } else if (val > max) {
        return max;
    } else {
        return val;
    }
}

Particle slowDown(Particle thisParticle) {
    thisParticle.velocity[0] *= 0.955;
    thisParticle.velocity[1] *= 0.955;
    thisParticle.velocity[2] *= 0.955;
    return thisParticle;
}

Particle speedUp(Particle thisParticle) {
    thisParticle.velocity[0] *= 1.1;
    thisParticle.velocity[1] *= 1.1;
    thisParticle.velocity[2] *= 1.1;
    return thisParticle;
}

kernel void fireworkComputeShader(
                                   device Particle *in [[ buffer(0) ]],
                                   device Particle *out [[ buffer(1) ]],
                                  const device SharedFireworkUniforms &uniforms [[ buffer(2) ]],
                                  const device SharedUniforms &sharedUniforms [[ buffer(3) ]],
                                   uint id [[thread_position_in_grid]])
{
    bool isHead = (id % 2 == 0);
    Particle thisParticle = in[id];
    Particle mouse;
    mouse.position = uniforms.mouse.position;
    mouse.mass = 1.5;
    
    if (isHead) {
        float3 diff;
        
        diff = thisParticle.position - (mouse.position);
        // diff = thisParticle.position - float3(0.0);

        float distance = constrain(length(diff), 10.0, 50.0);
        float strength = thisParticle.mass * mouse.mass / (distance * distance);

        diff = normalize(diff);
        diff = diff * strength * -0.083;
        
        thisParticle.velocity = thisParticle.velocity + diff * 0.78;
        thisParticle.position = thisParticle.position + thisParticle.velocity;
        
        if (distance > 15 || length(diff) > 0.5) {
            thisParticle = slowDown(thisParticle);
        } else {
            thisParticle = speedUp(thisParticle);
        }
    
    } else {
        Particle headParticle = in[id - 1];
        thisParticle.position = headParticle.position - headParticle.velocity * 2.0;
    }
    
    //mass
    out[id].position = thisParticle.position;
    out[id].velocity = thisParticle.velocity;
    out[id].mass = thisParticle.mass;
}


struct VertexOut {
    float4 position [[position]];
    float pointsize [[point_size]];
    float3 color;
};

vertex VertexOut particle_vertex(                           // 1
                                 device Particle *inParticle [[ buffer(0) ]], // 2
                                 constant SharedUniforms &uniforms [[ buffer(1) ]],
                                 unsigned int id [[ vertex_id ]]) {                 // 3
    
    
    float4x4 mv_Matrix = uniforms.viewMatrix;
    float4x4 proj_Matrix = uniforms.projectionMatrix;
    
    Particle thisParticle = inParticle[id];
    VertexOut VertexOut;
    
//    VertexOut.position = proj_Matrix * mv_Matrix * float4(thisParticle.position, 1.0);
    VertexOut.position = proj_Matrix * mv_Matrix * float4(thisParticle.position, 1.0);
    VertexOut.pointsize = 1.0;
    VertexOut.color = thisParticle.velocity;
    
    return VertexOut;              // 4
}


fragment half4 particle_fragment(
                                 VertexOut         interpolated       [[stage_in]]
                                 ) {
    
    interpolated.color *= 100.0;
    
    if (interpolated.color.x < 0.5) {
        interpolated.color.x += 0.35;
    }
    if (interpolated.color.y < 0.5) {
        interpolated.color.y += 0.35;
    }
    if (interpolated.color.z < 0.5) {
        interpolated.color.z += 0.35;
    }
    
    return half4(
                 interpolated.color.x,
                 interpolated.color.y,
                 interpolated.color.z, 1.0);
}

