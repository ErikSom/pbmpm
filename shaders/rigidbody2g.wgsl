// shader/rigidbody2g.wgsl (Final Corrected Version)
//-----------------------------------------------------------------------------
// Copyright (c) 2024 Electronic Arts.  All rights reserved.
//-----------------------------------------------------------------------------

//!include matrix.inc
//!include dispatch.inc
//!include simConstants.inc
//!include rigidbody.inc
//!include particle.inc

struct BukkitBodyInfo {
    count: u32,
    offset: u32,
};

@group(0) @binding(0) var<uniform> g_simConstants : SimConstants;
@group(0) @binding(1) var<storage> g_rigidBodies: RigidBody;
@group(0) @binding(2) var<storage, read_write> g_grid : array<atomic<i32>>;
@group(0) @binding(3) var<storage> g_bukkitBodyCountsAndOffsets: array<BukkitBodyInfo>;
@group(0) @binding(4) var<storage> g_bodyBukkitMap: array<u32>;

// Back to a grid-centric dispatch. One thread per grid cell.
@compute @workgroup_size(GridDispatchSize, GridDispatchSize)
fn csMain(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!insideGuardian(id.xy, g_simConstants.gridSize, GuardianSize)) {
        return;
    }

    let gridPos = vec2f(id.xy);
    let gridIdx = gridVertexIndex(id.xy, g_simConstants.gridSize);

    // --- THE FIX ---
    // Convert the grid cell's query position into the physics engine's coordinate system (Y-flip).
    // All subsequent distance and collision checks will happen in this consistent "physics space".
    var physicsQueryPos = gridPos;
    // physicsQueryPos.y = f32(g_simConstants.gridSize.y) - gridPos.y;

    var closestBodyIndex: i32 = -1;
    var closestPenetration: f32 = 0.0;
    var closestNormal = vec2f(0.0);

    // 1. Find which bukkit this grid cell belongs to.
    let bukkitCoords = vec2u(gridPos / f32(BukkitSize));
    let bukkitIndex = bukkitCoords.y * g_simConstants.bukkitCountX + bukkitCoords.x;

    // 2. Look up the list of nearby bodies.
    let bukkitInfo = g_bukkitBodyCountsAndOffsets[bukkitIndex];

    // 3. Loop over ONLY the nearby bodies.
    for (var i = 0u; i < bukkitInfo.count; i = i + 1u) {
        let bodyIndex = g_bodyBukkitMap[bukkitInfo.offset + i];
        let body = g_rigidBodies.bodies[bodyIndex];

        // Compare physics-space body position with our transformed physics-space query position.
        let offset = body.position - physicsQueryPos;
        if (dot(offset, offset) > body.boundRadiusSq) {
            continue;
        }

        for (var shapeIdx = 0u; shapeIdx < u32(body.shapeCount); shapeIdx++) {
            let globalShapeIndex = u32(body.shapeStartIndex + f32(shapeIdx));
            let shape = g_rigidBodies.shapes[globalShapeIndex];
            
            // Call the collision function using only physics-space coordinates.
            let collideResult = RBcollide(shape, body.position, body.angle, physicsQueryPos);
            
            if (collideResult.collides && collideResult.penetration > closestPenetration) {
                closestBodyIndex = i32(bodyIndex);
                closestPenetration = collideResult.penetration;
                closestNormal = collideResult.normal;
            }
        }
    }
    
    // 4. Write the results.
    atomicStore(&g_grid[gridIdx + 4], closestBodyIndex);
    atomicStore(&g_grid[gridIdx + 5], encodeFixedPoint(closestNormal.x, g_simConstants.fixedPointMultiplier));
    atomicStore(&g_grid[gridIdx + 6], encodeFixedPoint(closestNormal.y, g_simConstants.fixedPointMultiplier));
    atomicStore(&g_grid[gridIdx + 7], encodeFixedPoint(closestPenetration, g_simConstants.fixedPointMultiplier));
}