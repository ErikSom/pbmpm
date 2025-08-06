//-----------------------------------------------------------------------------
// Copyright (c) 2024 Electronic Arts.  All rights reserved.
//-----------------------------------------------------------------------------

//!include matrix.inc
//!include dispatch.inc
//!include simConstants.inc
//!include rigidbody.inc
//!include particle.inc

@group(0) @binding(0) var<uniform> g_simConstants : SimConstants;
@group(0) @binding(1) var<storage> g_rigidBodies: RigidBody;
@group(0) @binding(2) var<storage, read_write> g_grid : array<atomic<i32>>;

@compute @workgroup_size(GridDispatchSize, GridDispatchSize)
fn csMain(@builtin(global_invocation_id) id: vec3<u32>)
{
    if(!insideGuardian(id.xy, g_simConstants.gridSize, GuardianSize))
    {
        return;
    }
    
    let gridPos = vec2f(id.xy);
    let gridIdx = gridVertexIndex(id.xy, g_simConstants.gridSize);
    
    var closestBodyIndex = -1;
    var closestPenetration = 0.0;
    var closestNormal = vec2f(0.0);
    var closestPoint = vec2f(0.0);
    
    // Check all rigid bodies at this grid point
    for (var bodyIndex = 0u; bodyIndex < g_rigidBodies.body_count; bodyIndex++)
    {
        let body = g_rigidBodies.bodies[bodyIndex];
        
        for (var i = 0u; i < u32(body.shapeCount); i++)
        {
            let shapeIndex = u32(body.shapeStartIndex + f32(i));
            let shape = g_rigidBodies.shapes[shapeIndex];
            
            let collideResult = RBcollide(shape, body.position, body.angle, gridPos);
            
            if (collideResult.collides && collideResult.penetration > closestPenetration)
            {
                closestBodyIndex = i32(bodyIndex);
                closestPenetration = collideResult.penetration;
                closestNormal = collideResult.normal;
                closestPoint = collideResult.pointOnCollider;
            }
        }
    }
    
    // 4: body index (-1 if none)
    // 5: normal X
    // 6: normal Y
    // 7: penetration
    atomicStore(&g_grid[gridIdx + 4], closestBodyIndex);
    atomicStore(&g_grid[gridIdx + 5], encodeFixedPoint(closestNormal.x, g_simConstants.fixedPointMultiplier));
    atomicStore(&g_grid[gridIdx + 6], encodeFixedPoint(closestNormal.y, g_simConstants.fixedPointMultiplier));
    atomicStore(&g_grid[gridIdx + 7], encodeFixedPoint(closestPenetration, g_simConstants.fixedPointMultiplier));
}