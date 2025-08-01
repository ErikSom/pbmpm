//-----------------------------------------------------------------------------
// Copyright (c) 2024 Electronic Arts.  All rights reserved.
//-----------------------------------------------------------------------------

"use strict";

import * as shader from "./shader.js"
import { g_rigidBodyFactory } from "./sim.js";

let context = {
    pipelines: {},
    maxParticleCount: 1000000,

    maxTimeStampCount: 2048,

    encoder: null,
    frameTimeStampCount: 0,
    frameTimeStampNames: {},

    movingAverageTimeStamps: {},

    timingStatsDirty: false,
    particleCountDirty: false,

    particleCount: 0,
    particleFreeCount: 0,
};

export function getGpuContext() {return context;}

export function divUp(threadCount, divisor)
{
    return Math.floor((threadCount + divisor - 1) / divisor);
}

export function createBindGroup(name, shaderName, resources)
{
    let entries = [];
    for(let i = 0; i < resources.length; ++i)
    {
        entries.push({binding: i, resource: {buffer: resources[i]}});   
    }

    return context.device.createBindGroup({
        label: name,
        layout: context.pipelines[shaderName].getBindGroupLayout(0),
        entries: entries
    });
}

export function construct4IntBuffer(name, usage, values)
{
    const buf = context.device.createBuffer({
        name: name, 
        size: 16,
        usage: usage
    })

    const valueArray = new Int32Array(4);
    valueArray.set(values);
    context.device.queue.writeBuffer(buf, 0, valueArray);

    return buf;
}

export function resetBuffers(gridSize)
{
    // Construct various small buffers used for indirect dispatch, counting and staging
    context.particleCountBuffer = construct4IntBuffer('particleCountBuffer', GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, [0,0,0,0]);
    context.particleCountStagingBuffer = construct4IntBuffer('particleCountStagingBuffer', GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, [0,0,0,0]);
    context.particleRenderDispatchBuffer = construct4IntBuffer('particleRenderDispatchBuffer', GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT, [6,0,0,0]);
    context.particleSimDispatchBuffer = construct4IntBuffer('particleSimDispatchBuffer', GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, [0,1,1,0]);
    context.particleFreeCountStagingBuffer = construct4IntBuffer('particleFreeCountStagingBuffer', GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, [0,0,0,0]);

    const maxRigidBodies = 128;

    // Buffer to send Box2D body data to the GPU
    context.rigidBodiesBuffer = context.device.createBuffer({
        label: "rigidBodiesBuffer (CPU->GPU)",
        size: maxRigidBodies * g_rigidBodyFactory.getTotalSizeInWords() * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });


    // We use 4 floats per body: force.x, force.y, torque, and padding.
    const forceResultSize = 16; // 4 floats * 4 bytes
    context.forceResultsBuffer = context.device.createBuffer({
        label: "forceResultsBuffer (GPU->CPU)",
        size: maxRigidBodies * forceResultSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // ✨ NEW: Staging buffer to read force results back to the CPU without stalling
    context.forceResultsStagingBuffer = context.device.createBuffer({
        label: "forceResultsStagingBuffer",
        size: maxRigidBodies * forceResultSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Construct particle buffer.
    // Must be kept in sync with MPMParticle in particle.inc.wgsl
    const particleFloatCount = 24;
    context.particleBuffer = context.device.createBuffer({
        label: "particleBuffer",
        size: context.maxParticleCount * 4 * particleFloatCount,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    context.particleFreeIndicesBuffer = context.device.createBuffer({
        label: 'freeIndices',
        size: 4 + context.maxParticleCount * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });


}

export function beginFrame()
{
    context.frameTimeStampCount = 0;
    context.frameTimeStampNames = {};
    context.encoder = context.device.createCommandEncoder();
}

export function endFrame()
{
    const canReadbackParticleCount = context.particleCountStagingBuffer.mapState === 'unmapped';
    const canReadbackParticleFreeCount = context.particleFreeCountStagingBuffer.mapState === 'unmapped';
    const canReadbackTimeStamps = context.canTimeStamp && context.timeStampResultBuffer.mapState === 'unmapped';
    const canReadbackForces = context.forceResultsStagingBuffer.mapState === 'unmapped';
    
    if(canReadbackParticleCount)
    {
        context.encoder.copyBufferToBuffer(context.particleCountBuffer, 0, context.particleCountStagingBuffer, 0, 4);
    }

    if(canReadbackParticleFreeCount)
    {
        context.encoder.copyBufferToBuffer(context.particleFreeIndicesBuffer, 0, context.particleFreeCountStagingBuffer, 0, 4);
    }

    if(canReadbackTimeStamps)
    {
        context.encoder.resolveQuerySet(context.timeStampQuerySet, 0, context.frameTimeStampCount, context.timeStampResolveBuffer, 0);

        context.encoder.copyBufferToBuffer(context.timeStampResolveBuffer, 0, context.timeStampResultBuffer, 0, context.timeStampResultBuffer.size);
    }

    if (canReadbackForces)
    {
        context.encoder.copyBufferToBuffer(
            context.forceResultsBuffer, 0, 
            context.forceResultsStagingBuffer, 0, 
            context.forceResultsStagingBuffer.size
        );
    }

    context.device.queue.submit([context.encoder.finish()]);

    if(canReadbackParticleCount)
    {
        readbackParticleCount();
    }

    if(canReadbackParticleFreeCount)
    {
        readbackParticleFreeCount();
    }

    if(canReadbackTimeStamps)
    {
        readbackTimeStamps();
    }

    if(canReadbackForces) {
        readbackImpulses(context.lastFrameInputs, context.lastFrameRigidBodyData);
    }

    context.encoder = null;
}

// Helper function to dispatch a compute shader with the given name, resources
// and dispatch size.
// if GroupCount is an array then a regular dispatch is done, otherwise
// it is bound as an indirect buffer and an indirect dispatch is done.
export function computeDispatch(shaderName, resources, groupCount)
{
    // Construct array of resources in the required format for
    // creating a bind group
    let entries = []

    var isSimpleFlatBufferList = true;
    var isNestedBufferList = true;
    for(let i = 0; i < resources.length; ++i)
    {
        if(!resources[i])
        {
            throw `Compute Dispatch [${shaderName}]: Resource at index ${i} was falsy!`
        }
        if(!(resources[i] instanceof GPUBuffer))
        {
            isSimpleFlatBufferList = false;
        }
        if(Array.isArray(resources[i]))
        {
            for(let j = 0; j < resources[i].length; ++j)
            {
                if(!resources[i][j])
                {
                    throw `Compute Dispatch [${shaderName}]: Resource at group ${i} index ${j} was falsy!`
                }
                if(!(resources[i][j] instanceof GPUBuffer))
                {
                    isNestedBufferList = false;
                }
            }
        }
    }

    if(!isSimpleFlatBufferList && !isNestedBufferList)
    {
        throw `Expected resources to be an array of resources OR an array of arrays of resources.`;
    }

    if(isSimpleFlatBufferList)
    {
        entries.push([])
        for(let i = 0; i < resources.length; ++i)
        {
            entries[0].push({binding: i, resource: {buffer: resources[i]}});
        }
    }
    else if(isNestedBufferList)
    {
        for(let i = 0; i < resources.length; ++i)
        {
            let thisGroupEntries = []
    
            for(let j = 0; j < resources[i].length; ++j)
            {
                if(!resources[i][j])
                {
                    throw `Compute Dispatch [${shaderName}]: Resource at group ${i} index ${j} was falsy!`
                }
        
                thisGroupEntries.push({binding: j, resource: {buffer: resources[i][j]}});
            }
    
            entries.push(thisGroupEntries);
        }
    }

    const pipeline = shader.getComputePipeline(shaderName);
    
    const computePass = context.encoder.beginComputePass({
        label: shaderName,
        ...(context.canTimeStamp && {
            timestampWrites: {
                querySet: context.timeStampQuerySet,
                beginningOfPassWriteIndex: context.frameTimeStampCount,
                endOfPassWriteIndex: context.frameTimeStampCount + 1
            }
        })
    });
    computePass.setPipeline(pipeline);

    for(let i = 0; i < entries.length; ++i)
    {
        const bindGroup = context.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: entries[i]});
        computePass.setBindGroup(i, bindGroup);
    }

    if(Array.isArray(groupCount))
    {
        computePass.dispatchWorkgroups(groupCount[0], groupCount[1], groupCount[2]);
    }
    else
    {
        computePass.dispatchWorkgroupsIndirect(groupCount, 0);
    }
    computePass.end()

    if(context.canTimeStamp)
    {
        if(shaderName in context.frameTimeStampNames)
        {
            context.frameTimeStampNames[shaderName].push(context.frameTimeStampCount);
        }
        else
        {
            context.frameTimeStampNames[shaderName] = [context.frameTimeStampCount];
        }

        context.frameTimeStampCount += 2;
    }
}

// Try to initialize webgpu device and set up the basic objects.
// This may fail on some browsers like Firefox, in which case we put a message in the dom.
// It may also fail if the browser has blacklisted webgpu due to previously going OOM.
// If this happens then it may be necessary to restart the whole program.
export async function init(insertHandlers)
{
    context.insertHandlers = insertHandlers;

    // Initialize device
    if (!navigator.gpu) {
        throw "WebGPU not supported on this browser.";
    }

    context.adapter = await navigator.gpu.requestAdapter();
    if (!context.adapter) {
        throw "No appropriate GPUAdapter found.";
    }

    context.canTimeStamp = context.adapter.features.has('timestamp-query');

    if(!context.canTimeStamp)
    {
        console.warn('This WebGPU implementation does not support timestamp queries. Timing info will not be available.');
    }

    context.device = await context.adapter.requestDevice({
        requiredFeatures: context.canTimeStamp ? [
             ['timestamp-query']
        ] : undefined,
        requiredLimits: {
            maxStorageBuffersPerShaderStage: 10
        }
    });
    
    await shader.init(context.device, insertHandlers);

    // Set back buffer pixel format
    context.context = document.getElementById('canvas').getContext("webgpu", {alpha: true});
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.context.configure({
        device: context.device,
        format: canvasFormat,
        alphaMode: 'premultiplied'
    });

    // Load the particle rendering module
    const renderShaderModule = shader.getShaderModule(shader.Shaders.particleRender);

    // Construct pipeline for particle rendering
    context.pipelines['particleRender'] = context.device.createRenderPipeline({
        label: "Render Pipeline",
        layout: "auto",
        vertex: {
            module: renderShaderModule,
            entryPoint: "vertexMain",
            buffers: []
        },
        fragment: {
            module: renderShaderModule,
            entryPoint: "fragmentMain",
            targets: [{
            format: canvasFormat,
            blend: {
                alpha: {
                    dstFactor: 'one-minus-src-alpha',
                    srcFactor: 'src-alpha',
                    operation: 'add'
                },
                color: {
                    dstFactor: 'one-minus-src-alpha',
                    srcFactor: 'src-alpha',
                    operation: 'add'
                }
            }
            }]
        },
    });    

    if(context.canTimeStamp)
    {
        context.timeStampQuerySet = context.device.createQuerySet({
            type: 'timestamp',
            count: context.maxTimeStampCount*2 // Begin and end
        });

        context.timeStampResolveBuffer = context.device.createBuffer({
            size: context.timeStampQuerySet.count * 8, // Timestamps are uint64
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
        });

        context.timeStampResultBuffer = context.device.createBuffer({
            size: context.timeStampQuerySet.count * 8,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
    }
}

// Initiate readback of the particle count staging buffer.
function readbackParticleCount()
{
    context.particleCountStagingBuffer.mapAsync(GPUMapMode.READ, 0, 16).then(() => {
        try
        {
            const buf = context.particleCountStagingBuffer.getMappedRange(0, 16);
            const view = new Int32Array(buf);
            context.particleCount = view[0];
            context.particleCount = Math.min(context.maxParticleCount, context.particleCount);
            context.particleCountDirty = true;
        }
        catch(e)
        {
    
        }
        finally
        {
            context.particleCountStagingBuffer.unmap();
        }
    }).catch(() => {});
}

function readbackParticleFreeCount()
{
    context.particleFreeCountStagingBuffer.mapAsync(GPUMapMode.READ, 0, 16).then(() => {
        try
        {
            const buf = context.particleFreeCountStagingBuffer.getMappedRange(0, 16);
            const view = new Int32Array(buf);
            context.particleFreeCount = view[0];
            context.particleCountDirty = true;
        }
        catch(e)
        {
    
        }
        finally
        {
            context.particleFreeCountStagingBuffer.unmap();
        }
    }).catch(() => {});
}

function readbackTimeStamps()
{
    context.timeStampResultBuffer.mapAsync(GPUMapMode.READ).then(() => {
        try
        {
            const times = new BigInt64Array(context.timeStampResultBuffer.getMappedRange());

            var movingAverageTimeStamps = {};
    
    
            for(const name of Object.keys(context.frameTimeStampNames))
            {
                var total = 0;
                for(const index of context.frameTimeStampNames[name])
                {
                    total += Number(times[index+1] - times[index]);
                }
    
                movingAverageTimeStamps[name] = total;
            }
    
            for(const name of Object.keys(movingAverageTimeStamps))
            {
                if(name in context.movingAverageTimeStamps)
                {
                    movingAverageTimeStamps[name] = 0.01*movingAverageTimeStamps[name] + 0.99*context.movingAverageTimeStamps[name]
                }
            }
    
            context.movingAverageTimeStamps = movingAverageTimeStamps;
    
            context.timingStatsDirty = true;
        }
        catch(e)
        {
    
        }
        finally
        {
            context.timeStampResultBuffer.unmap();
        }
    }).catch(() => {});
}

function readbackImpulses(inputs, bodyData) {
    const staging = context.forceResultsStagingBuffer;
    if (!staging || staging.mapState !== 'unmapped') return;

    context.device.queue.onSubmittedWorkDone().then(() => {
        // Bail if buffers were reset/recreated since we queued this.
        if (
            staging !== context.forceResultsStagingBuffer ||
            staging.mapState !== 'unmapped') {
            return;
        }

        staging.mapAsync(GPUMapMode.READ).then(() => {
            let dataCopy;
            try {
                // Use the same object we mapped.
                const mappedRange = staging.getMappedRange();
                // Copy out immediately so we can unmap right away.
                dataCopy = new Int32Array(mappedRange.slice(0));
            } catch (err) {
                // Make sure we leave the buffer unmapped on error.
                try { staging.unmap(); } catch {}
                console.error("Impulse readback failed during getMappedRange.", err);
                return;
            }

            try { staging.unmap(); } catch {}

            // Now, safely process the data we copied (CPU-side).
            if (!inputs || !bodyData) {
                console.warn("Skipping impulse application due to missing inputs/bodyData for this frame.");
                return;
            }

            // This value must match the 'forceMultiplier' (now impulseMultiplier) in sim.js
            const impulseMultiplier = 1000.0;
            const wordsPerBody = 4; // ix, iy, angular_impulse, padding
            const maxBodiesFromBuffer = Math.floor(dataCopy.length / wordsPerBody);
            const bodyCount = Math.min(bodyData.length, maxBodiesFromBuffer);

            const impulsesToApply = [];

            for (let i = 0; i < bodyCount; i++) {
                const offset = i * wordsPerBody;
                const ix = dataCopy[offset + 0] / impulseMultiplier;
                const iy = dataCopy[offset + 1] / impulseMultiplier;
                const angularImpulse = dataCopy[offset + 2] / impulseMultiplier;

                if (ix !== 0 || iy !== 0 || angularImpulse !== 0) {
                    impulsesToApply.push({ bodyIndex: i, impulse: { x: ix, y: iy }, angularImpulse });
                }
            }

            // This function should now call Box2D's ApplyLinearImpulse and ApplyAngularImpulse
            if (window.applyImpulses && impulsesToApply.length > 0) {
                try { window.applyImpulses(inputs, impulsesToApply); } catch (e) {
                    console.error("applyImpulses threw:", e);
                }
            }
        }).catch((e) => {
            console.error("mapAsync failed for impulse readback. This can happen if the GPU device is lost.", e);
        });
    });
}
