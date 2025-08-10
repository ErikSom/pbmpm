//-----------------------------------------------------------------------------
// Copyright (c) 2024 Electronic Arts.  All rights reserved.
//-----------------------------------------------------------------------------

"use strict";

export const f32 = 'f32';
export const vec2f = 'vec2f';
export const vec3f = 'vec3f';
export const u32 = 'u32';
export const vec2u = 'vec2u';

export const Storage = 'storage';
export const Uniform = 'uniform';

// This class allows us to conveniently code-generate structs to be used as uniform buffers
// and copy data to the gpu, including respecting packing rules, in order to avoid
// having to do a lot of boilerplate for every parameter we want to pass.
export class BufferFactory {
    constructor(name, mode) {
        this.name = name;
        this.mode = mode;
        this.elements = []
        this.compiled = false;
        this.paddingCount = 0;
        this.totalCount = 0;
        this.maxAlignment = 1;
    }

    // Add a parameter with the given name and type.
    // The name is used as a key for binding data on the js side and
    // also in the generated wgsl code.
    add(name, type, count = 1) {
        console.assert(!this.compiled);

        const isArray = count > 1;

        const requiredAlignment = getAlignment(type);

        this.maxAlignment = Math.max(this.maxAlignment, requiredAlignment);

        const elementSize = getSize(type);
        const elementStride = Math.ceil(elementSize / requiredAlignment) * requiredAlignment;

        const requiredSlotCount = elementStride * count;

        let alignmentAdjustment = this.totalCount % requiredAlignment;
        while (alignmentAdjustment !== 0) {
            // Add a padding element to align up the member to the correct address
            this.add(`padding${this.paddingCount}`, f32);
            this.paddingCount += 1;
            alignmentAdjustment -= 1;
        }

        this.elements.push({
            name: name,
            type: type,
            value: undefined,
            offset: this.totalCount,
            slotCount: requiredSlotCount,
            elementStride: elementStride,
            isArray: isArray,
            count: count
        });

        this.totalCount += requiredSlotCount;
    }

    // Finalize the factory after parameters have been added
    compile() {
        console.assert(!this.compiled);

        // Add final padding so the struct's total size is a multiple of its largest member's alignment.
        // This is crucial for arrays of structs in storage buffers.
        const requiredPadding = (this.maxAlignment - (this.totalCount % this.maxAlignment)) % this.maxAlignment;
        for (let i = 0; i < requiredPadding; i++) {
            const padName = `padding${this.paddingCount++}`;
            this.elements.push({ name: padName, type: f32, offset: this.totalCount++, isArray: false });
        }

        if (this.mode == Uniform) {
            // Round up uniform buffer size to a multiple of 16 bytes (4 words)
            const alignment = 4;
            this.totalCount = Math.ceil(this.totalCount / alignment) * alignment;
        }

        this.compiled = true;
    }

    // Assemble text that can be pasted into the const buffer definition
    getShaderText() {
        console.assert(this.compiled);

        let shaderText = `struct ${this.name}\n{\n`
        for (const elem of this.elements) {
            // Don't generate shader code for padding helpers
            if (elem.name.startsWith('padding')) continue;

            if (elem.isArray) {
                shaderText += `${elem.name}: array<${elem.type}, ${elem.count}>,\n`;
            } else {
                shaderText += `${elem.name}: ${elem.type},\n`;
            }
        }
        shaderText += "};\n";

        return shaderText;
    }

    getTotalSizeInWords() {
        console.assert(this.compiled);
        return this.totalCount;
    }

    // Build a uniform buffer object containing the currently stored values
    constructUniformBuffer(device, values) {
        console.assert(this.compiled);
        console.assert(this.mode == Uniform);
        console.assert(Array.isArray(values));

        for (const elem of this.elements) {
            elem.value = undefined;
        }

        for (const valueObject of values) {
            for (const elem of this.elements) {
                if (elem.name in valueObject) {
                    elem.value = valueObject[elem.name];
                }
            }
        }

        for (const elem of this.elements) {
            if (elem.value === undefined && elem.name.indexOf('padding') == -1) {
                throw `Element ${elem.name} has never had its value set.`;
            }
        }

        const uniformBuffer = device.createBuffer({
            label: this.name,
            size: this.totalCount * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const uniformValues = new Float32Array(this.totalCount);

        for (const elem of this.elements) {
            if (elem.value === undefined) {
                continue;
            }

            if (elem.isArray) {
                const elementSize = getSize(elem.type); // Size of the base type (e.g., vec3f -> 3)
                for (let i = 0; i < elem.count; ++i) {
                    // The destination offset respects the stride (padding)
                    const destOffset = elem.offset + i * elem.elementStride;
                    // The source offset is tightly packed
                    const sourceOffset = i * elementSize;
                    const singleElementValue = elem.value.slice(sourceOffset, sourceOffset + elementSize);

                    if (elem.type == u32) {
                        const castArray = new Int32Array(1);
                        castArray.set(singleElementValue, 0);
                        const castArrayFloat = new Float32Array(castArray.buffer);
                        uniformValues.set(castArrayFloat, destOffset);
                    } else if (elem.type == vec2u) {
                        const castArray = new Int32Array(2);
                        castArray.set(singleElementValue, 0);
                        const castArrayFloat = new Float32Array(castArray.buffer);
                        uniformValues.set(castArrayFloat, destOffset);
                    } else {
                        // f32, vec2f, vec3f can be set directly
                        uniformValues.set(singleElementValue, destOffset);
                    }
                }
            } else {
                // If we need to write an integer type then we have to trick
                // the data into the uniform values array using this mechanism.
                if (elem.type == u32) {
                    const castArray = new Int32Array(1);
                    castArray.set([elem.value], 0);
                    const castArrayFloat = new Float32Array(castArray.buffer);
                    uniformValues.set(castArrayFloat, elem.offset);
                }
                else if (elem.type == vec2u) {
                    const castArray = new Int32Array(2);
                    castArray.set(elem.value, 0);
                    const castArrayFloat = new Float32Array(castArray.buffer);
                    uniformValues.set(castArrayFloat, elem.offset);
                }
                else if (elem.type == f32) {
                    // Float values can be written through an array
                    uniformValues.set([elem.value], elem.offset);
                }
                else {
                    // Float vector values can be written directly
                    uniformValues.set(elem.value, elem.offset);
                }
            }
        }

        device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

        return uniformBuffer;
    }

    constructStorageBuffer(device, elements) {
        console.assert(this.compiled);
        console.assert(this.mode == Storage);
        console.assert(Array.isArray(elements));

        // Note - no check that all data is properly set
        const elementCount = elements.length;

        const storageBuffer = device.createBuffer({
            label: this.name,
            size: this.totalCount * elementCount * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        const storageValues = new Float32Array(this.totalCount * elementCount);

        for (var i = 0; i < elementCount; ++i) {
            const outputOffset = i * this.totalCount;
            const currentElementData = elements[i];

            for (const elem of this.elements) {
                elem.value = undefined;
            }

            for (const elem of this.elements) {
                if (elem.name in elements[i]) {
                    elem.value = elements[i][elem.name];
                }
            }

            for (const elem of this.elements) {
                if (elem.value === undefined && elem.name.indexOf('padding') == -1) {
                    throw `Element ${elem.name} has never had its value set.`;
                }
            }

            for (const elem of this.elements) {
                const value = currentElementData[elem.name];

                if (value === undefined) {
                    if (elem.name.indexOf('padding') == -1) {
                        throw `Element ${elem.name} has no value set for index ${i}.`;
                    }
                    continue;
                }

                // Handle array types
                if (elem.isArray) {
                    const elementSize = getSize(elem.type);
                    for (let j = 0; j < elem.count; ++j) {
                        const destOffset = outputOffset + elem.offset + j * elem.elementStride;
                        const sourceOffset = j * elementSize;
                        const singleElementValue = value.slice(sourceOffset, sourceOffset + elementSize);

                        if (elem.type == u32) {
                            const castArray = new Int32Array(1);
                            castArray.set(singleElementValue, 0);
                            const castArrayFloat = new Float32Array(castArray.buffer);
                            storageValues.set(castArrayFloat, destOffset);
                        } else if (elem.type == vec2u) {
                            const castArray = new Int32Array(2);
                            castArray.set(singleElementValue, 0);
                            const castArrayFloat = new Float32Array(castArray.buffer);
                            storageValues.set(castArrayFloat, destOffset);
                        } else {
                            storageValues.set(singleElementValue, destOffset);
                        }
                    }
                } else {
                    // If we need to write an integer type then we have to trick
                    // the data into the uniform values array using this mechanism.
                    const destOffset = outputOffset + elem.offset;
                    if (elem.type == u32) {
                        const castArray = new Int32Array(1);
                        castArray.set([value], 0);
                        const castArrayFloat = new Float32Array(castArray.buffer);
                        storageValues.set(castArrayFloat, destOffset);
                    } else if (elem.type == vec2u) {
                        const castArray = new Int32Array(2);
                        castArray.set(value, 0);
                        const castArrayFloat = new Float32Array(castArray.buffer);
                        storageValues.set(castArrayFloat, destOffset);
                    } else if (elem.type == f32) {
                        storageValues.set([value], destOffset);
                    } else {
                        storageValues.set(value, destOffset);
                    }
                }
            }
        }

        device.queue.writeBuffer(storageBuffer, 0, storageValues);

        return storageBuffer;
    }

    constructCPUArray(elements) {
        console.assert(this.compiled);
        console.assert(Array.isArray(elements));

        const elementCount = elements.length;
        const cpuValues = new Float32Array(this.totalCount * elementCount);

        for (var i = 0; i < elementCount; ++i) {
            const outputOffset = i * this.totalCount;
            const currentElementData = elements[i];

            for (const elem of this.elements) {
                const value = currentElementData[elem.name];

                if (value === undefined) {
                    continue;
                }

                // Handle array types
                if (elem.isArray) {
                    const elementSize = getSize(elem.type);
                    for (let j = 0; j < elem.count; ++j) {
                        const destOffset = outputOffset + elem.offset + j * elem.elementStride;
                        const sourceOffset = j * elementSize;
                        const singleElementValue = value.slice(sourceOffset, sourceOffset + elementSize);

                        if (elem.type == u32) {
                            const castArray = new Int32Array(1);
                            castArray.set(singleElementValue, 0);
                            const castArrayFloat = new Float32Array(castArray.buffer);
                            cpuValues.set(castArrayFloat, destOffset);
                        } else if (elem.type == vec2u) {
                            const castArray = new Int32Array(2);
                            castArray.set(singleElementValue, 0);
                            const castArrayFloat = new Float32Array(castArray.buffer);
                            cpuValues.set(castArrayFloat, destOffset);
                        } else {
                            cpuValues.set(singleElementValue, destOffset);
                        }
                    }
                }
                // Handle scalar/vector types
                else {
                    const destOffset = outputOffset + elem.offset;
                    if (elem.type == u32) {
                        const castArray = new Int32Array(1);
                        castArray.set([value], 0);
                        const castArrayFloat = new Float32Array(castArray.buffer);
                        cpuValues.set(castArrayFloat, destOffset);
                    }
                    else if (elem.type == vec2u) {
                        const castArray = new Int32Array(2);
                        castArray.set(value, 0);
                        const castArrayFloat = new Float32Array(castArray.buffer);
                        cpuValues.set(castArrayFloat, destOffset);
                    }
                    else if (elem.type == f32) {
                        cpuValues.set([value], destOffset);
                    }
                    else {
                        cpuValues.set(value, destOffset);
                    }
                }
            }
        }
        return cpuValues;
    }

    getUniformData(values) {
        console.assert(this.compiled);
        console.assert(this.mode == Uniform);
        console.assert(Array.isArray(values));

        for (const elem of this.elements) {
            elem.value = undefined;
        }

        for (const valueObject of values) {
            for (const elem of this.elements) {
                if (elem.name in valueObject) {
                    elem.value = valueObject[elem.name];
                }
            }
        }

        for (const elem of this.elements) {
            if (elem.value === undefined && elem.name.indexOf('padding') == -1) {
                throw `Element ${elem.name} has never had its value set.`;
            }
        }

        const uniformValues = new Float32Array(this.totalCount);

        for (const elem of this.elements) {
            if (elem.value === undefined) {
                continue;
            }

            if (elem.isArray) {
                const elementSize = getSize(elem.type); // Size of the base type (e.g., vec3f -> 3)
                for (let i = 0; i < elem.count; ++i) {
                    const destOffset = elem.offset + i * elem.elementStride;
                    const sourceOffset = i * elementSize;
                    const singleElementValue = elem.value.slice(sourceOffset, sourceOffset + elementSize);

                    if (elem.type == u32) {
                        const castArray = new Int32Array(1);
                        castArray.set(singleElementValue, 0);
                        const castArrayFloat = new Float32Array(castArray.buffer);
                        uniformValues.set(castArrayFloat, destOffset);
                    } else if (elem.type == vec2u) {
                        const castArray = new Int32Array(2);
                        castArray.set(singleElementValue, 0);
                        const castArrayFloat = new Float32Array(castArray.buffer);
                        uniformValues.set(castArrayFloat, destOffset);
                    } else {
                        uniformValues.set(singleElementValue, destOffset);
                    }
                }
            } else {
                if (elem.type == u32) {
                    const castArray = new Int32Array(1);
                    castArray.set([elem.value], 0);
                    const castArrayFloat = new Float32Array(castArray.buffer);
                    uniformValues.set(castArrayFloat, elem.offset);
                }
                else if (elem.type == vec2u) {
                    const castArray = new Int32Array(2);
                    castArray.set(elem.value, 0);
                    const castArrayFloat = new Float32Array(castArray.buffer);
                    uniformValues.set(castArrayFloat, elem.offset);
                }
                else if (elem.type == f32) {
                    uniformValues.set([elem.value], elem.offset);
                }
                else {
                    uniformValues.set(elem.value, elem.offset);
                }
            }
        }

        return uniformValues;
    }
}

// What should the size of each type be in multiples of the size of
// an f32
function getSize(type) {
    switch (type) {
        case f32:
        case u32:
            return 1;
        case vec2f:
        case vec2u:
            return 2;
        case vec3f:
            return 3;
        default:
            throw `Unsupported type [${type}]`;
    }
}

// What should the alignment of each type be in multiples of
// the size of an f32
function getAlignment(type) {
    switch (type) {
        case f32:
        case u32:
            return 1;
        case vec2f:
        case vec2u:
            return 2;
        case vec3f:
            return 4;
        default:
            throw `Unsupported type [${type}]`;
    }
}
