export class RigidBodyBufferHandler {
    constructor(name, bodyDefFactory, shapeFactory, maxBodies, maxShapes) {
        this.name = name;
        this.bodyDefFactory = bodyDefFactory;
        this.shapeFactory = shapeFactory;
        this.maxBodies = maxBodies;
        this.maxShapes = maxShapes;

        this.bodyStrideBytes = bodyDefFactory.getTotalSizeInWords() * 4;
        this.shapeStrideBytes = shapeFactory.getTotalSizeInWords() * 4;

        this.headerSizeBytes = 16; // 4 words * 4 bytes/word (for body_count, shape_count, padding)
        this.bodiesOffsetBytes = this.headerSizeBytes;
        this.shapesOffsetBytes = this.bodiesOffsetBytes + (this.bodyStrideBytes * this.maxBodies);
        this.totalSizeInBytes = this.shapesOffsetBytes + (this.shapeStrideBytes * this.maxShapes);
    }

    getShaderText() {
        const bodyDefShaderText = this.bodyDefFactory.getShaderText();
        const shapeShaderText = this.shapeFactory.getShaderText();

        return `
${bodyDefShaderText}
${shapeShaderText}

const MAX_BODIES = ${this.maxBodies}u;
const MAX_SHAPES = ${this.maxShapes}u;

struct ${this.name} {
    body_count: u32,
    shape_count: u32,
    padding: vec2f,
    bodies: array<BodyDef, MAX_BODIES>,
    shapes: array<Shape, MAX_SHAPES>,
};
        `;
    }

    getTotalSizeInWords() {
        return (this.totalSizeInBytes / 4);
    }

    constructCPUArray(sceneData) {
        console.assert(sceneData.bodies.length <= this.maxBodies, "Exceeded MAX_BODIES");
        console.assert(sceneData.shapes.length <= this.maxShapes, "Exceeded MAX_SHAPES");

        const cpuBuffer = new ArrayBuffer(this.totalSizeInBytes);
        const cpuFloat32View = new Float32Array(cpuBuffer);
        const cpuUint32View = new Uint32Array(cpuBuffer);

        cpuUint32View[0] = sceneData.bodies.length;
        cpuUint32View[1] = sceneData.shapes.length;

        const bodiesArray = this.bodyDefFactory.constructCPUArray(sceneData.bodies);
        const shapesArray = this.shapeFactory.constructCPUArray(sceneData.shapes);

        cpuFloat32View.set(bodiesArray, this.bodiesOffsetBytes / 4);
        cpuFloat32View.set(shapesArray, this.shapesOffsetBytes / 4);

        return cpuFloat32View;
    }
}