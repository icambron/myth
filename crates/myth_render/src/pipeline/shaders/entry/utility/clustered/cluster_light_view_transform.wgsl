// Pre-transforms local lights into view-space bounding spheres once per frame.
// The .w component stores a conservative effective culling radius derived from
// the runtime distance attenuation curve, so cluster_cull only has to perform
// pure geometric sphere-vs-frustum tests.

{{ binding_code }}
{{ scene_lighting_structs }}

@group(1) @binding(0) var<uniform> u_local_light_buffer_metadata: LightBufferMetadata;
@group(1) @binding(1) var<storage, read> st_local_lights: array<Struct_lights>;
@group(1) @binding(2) var<storage, read_write> st_light_view_positions: array<vec4<f32>>;

const LIGHT_VIEW_TRANSFORM_WG_SIZE: u32 = 64u;
const LIGHT_INTENSITY_CULL_THRESHOLD: f32 = 0.005;
const SPOT_TIGHT_SPHERE_COS_THRESHOLD: f32 = 0.70710678;

{$ include 'core/light_attenuation' $}

fn solve_effective_light_distance(light: Struct_lights) -> f32 {
    return getLightCullingDistance(
        light.intensity, light.range, light.decay, LIGHT_INTENSITY_CULL_THRESHOLD,
    );
}

fn build_spot_bounding_sphere(light: Struct_lights, effective_range: f32) -> vec4<f32> {
    if (light.outer_cone_cos < SPOT_TIGHT_SPHERE_COS_THRESHOLD) {
        let view_pos = (u_render_state.view_matrix * vec4<f32>(light.position, 1.0)).xyz;
        return vec4<f32>(view_pos, effective_range);
    }

    let cos_val = max(light.outer_cone_cos, 1e-4);
    let radius = 0.5 * effective_range / cos_val;
    let sphere_center = light.position + light.direction * radius;
    let sphere_center_view = (u_render_state.view_matrix * vec4<f32>(sphere_center, 1.0)).xyz;
    return vec4<f32>(sphere_center_view, radius);
}

@compute @workgroup_size(LIGHT_VIEW_TRANSFORM_WG_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let light_index = global_id.x;
    let local_light_count = min(
        u_local_light_buffer_metadata.total_light_count,
        min(arrayLength(&st_local_lights), arrayLength(&st_light_view_positions)),
    );
    if light_index >= local_light_count {
        return;
    }

    let light = st_local_lights[light_index];
    let effective_range = solve_effective_light_distance(light);
    if (effective_range <= 0.0) {
        st_light_view_positions[light_index] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        return;
    }

    if (light.light_type == 2u) {
        st_light_view_positions[light_index] = build_spot_bounding_sphere(light, effective_range);
        return;
    }

    let view_pos = (u_render_state.view_matrix * vec4<f32>(light.position, 1.0)).xyz;
    st_light_view_positions[light_index] = vec4<f32>(view_pos, effective_range);
}