//!insert RigidBody

struct RBCollideResult
{
    collides: bool,
    penetration: f32,
    normal: vec2f,
    pointOnCollider: vec2f,
};

fn RBcollide(shape: Shape, body_pos: vec2f, body_rot: f32, query_pos: vec2f) -> RBCollideResult
{
    
    let R_body = rot(body_rot);
    let world_shape_pos = body_pos + R_body * shape.position;

    if(shape.shapeType == B2CircleShape)
    {
        let offset = world_shape_pos - query_pos;
        let offsetLen = length(offset);
        
        if (offsetLen > shape.radius) {
             return RBCollideResult(false, 0.0, vec2f(0,0), vec2f(0,0));
        }

        let normal = offset * select(1.0/offsetLen, 0.0, offsetLen == 0.0);
        
        return RBCollideResult(
            true,
            shape.radius - offsetLen,
            normal,
            world_shape_pos - normal * shape.radius,
        );
    }
    else if(shape.shapeType == B2PolygonShape)
    {
        // R_body is the local-to-world transform, defined at the start of the RBcollide function
        let v = query_pos - world_shape_pos;

        // Early exit with bounding circle
        let r2 = shape.boundRadius * shape.boundRadius;
        if (dot(v,v) > r2) {
            return RBCollideResult(false, 0.0, vec2f(0,0), vec2f(0,0));
        }

        // Transform query point into shape's local space
        let RT = transpose(R_body);
        let pLocal = RT * v;

        // Find axis of least penetration in local space (SAT)
        var bestD : f32 = -3.402823e38;
        var bestI : i32 = -1;

        for (var i : i32 = 0; i < i32(shape.polyCount); i = i + 1) {
            let d = dot(shape.polyNormals[i], pLocal) - shape.polyPlane[i];
            if (d > 0.0) {
                // Found a separating axis, no collision
                return RBCollideResult(false, 0.0, vec2f(0,0), vec2f(0,0));
            }
            if (d > bestD) {
                bestD = d;
                bestI = i;
            }
        }

        // Calculate penetration and closest point on shape's boundary IN LOCAL SPACE
        let penetration = -bestD;
        let a = shape.polyVerts[bestI];
        let nextI = (bestI + 1) % i32(shape.polyCount);
        let b = shape.polyVerts[nextI];
        let e = b - a;
        let proj = pLocal - shape.polyNormals[bestI] * bestD;
        let t = clamp( dot(proj - a, e) / dot(e, e), 0.0, 1.0 );
        let closestLocal = a + e * t;

        // --- FIX IS HERE ---
        // Transform results from local space back to world space using the correct matrix (R_body)
        let inwardN = -(R_body * shape.polyNormals[bestI]);
        let pointOnPoly = world_shape_pos + R_body * closestLocal;

        return RBCollideResult(
            true,
            penetration,
            inwardN,
            pointOnPoly
        );
    }
    else
    {
        return RBCollideResult(false, 0.0, vec2f(0,0), vec2f(0,0));
    }
}