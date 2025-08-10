//-----------------------------------------------------------------------------
// Copyright (c) 2024 Electronic Arts.  All rights reserved.
//-----------------------------------------------------------------------------

"use strict";

import * as gpu from "./gpu.js"
import * as buffer_factory from "./buffer_factory.js"
import { RenderEnums } from "./sim.js";

let g_renderFactory;
let g_renderUniformBuffer = null;

export function init(insertHandlers) {
    // Specify the contents of the render uniform buffer.
    const renderFactory = new buffer_factory.BufferFactory('RenderConstants', buffer_factory.Uniform);
    renderFactory.add('particleRadiusTimestamp', buffer_factory.vec2f);
    renderFactory.add('canvasSize', buffer_factory.vec2f);

    renderFactory.add('viewPos', buffer_factory.vec2f);
    renderFactory.add('viewExtent', buffer_factory.vec2f);

    renderFactory.add('renderMode', buffer_factory.f32);
    renderFactory.add('deltaTime', buffer_factory.f32);
    renderFactory.compile();

    insertHandlers[renderFactory.name] = renderFactory.getShaderText();

    g_renderFactory = renderFactory;
}

export function update(gpuContext, inputs) {
    let viewPos = [inputs.gridSize[0] / 2, inputs.gridSize[1] / 2];
    let viewExtent = viewPos;

    const setDirectlyValues = {
        particleRadiusTimestamp: [0.5, 0],
        canvasSize: inputs.resolution,
        viewPos: viewPos,
        viewExtent: viewExtent,
        deltaTime: 1.0 / inputs.simRate,
    };

    const uniformData = g_renderFactory.getUniformData([inputs, setDirectlyValues]);

    if (!g_renderUniformBuffer) {
        g_renderUniformBuffer = gpuContext.device.createBuffer({
            label: 'RenderConstants',
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    gpuContext.device.queue.writeBuffer(g_renderUniformBuffer, 0, uniformData);

    const renderingBindGroup = gpu.createBindGroup(
        "Rendering Bind Group",
        'particleRender',
        [
            g_renderUniformBuffer,
            gpuContext.particleBuffer,
            gpuContext.rigidBodiesBuffer,
        ]
    );

    const renderPass = gpuContext.encoder.beginRenderPass({
        colorAttachments: [{
            view: gpuContext.context.getCurrentTexture().createView(),
            clearValue: [0, 0, 0, 0],
            loadOp: "clear",
            storeOp: "store",
        }]
    });

    renderPass.setPipeline(gpuContext.pipelines['particleRender']);
    renderPass.setBindGroup(0, renderingBindGroup);

    if (parseInt(inputs.renderMode) === RenderEnums.RenderModeRigidBody) {
        renderPass.draw(6, 1, 0, 0);
    } else {
        renderPass.drawIndirect(gpuContext.particleRenderDispatchBuffer, 0);
    }

    renderPass.end();
}