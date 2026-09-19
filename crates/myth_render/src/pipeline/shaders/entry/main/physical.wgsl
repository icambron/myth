// ── PBR Physical Material Entry Point ────────────────────────────────────
//
// Forward-rendered physically-based material with Cook-Torrance GGX BRDF.
// Supports: IBL, transmission, clearcoat, iridescence, sheen, anisotropy,
// SSAO integration, debug view overrides, and MRT specular split (SSSS).

{{ vertex_input_code }} 
{{ binding_code }}
{{ scene_lighting_structs }}
$$ if USE_CLUSTERED_SHADING is defined
{{ clustered_lighting_structs }}
$$ endif
{$ include 'core/vertex_output' $}
{$ include 'core/fragment_output' $}

{$ include 'modules/geometry/morphing' $}
{$ include 'modules/geometry/skinning' $}
{$ include 'core/common' $}
{$ include 'modules/lighting/punctual' $}
{$ include 'modules/bsdf/physical' $}

{$ include 'modules/lighting/iridescence' $}

$$ if USE_TRANSMISSION is defined
    {$ include 'modules/lighting/transmission' $}
$$ endif

{$ include 'core/alpha_test' $}
{$ include 'modules/bsdf/pbr_tone_mapping' $}

// The stock tangent-frame helper is emitted only for normal-map, clearcoat,
// and anisotropy variants. Height can be present by itself, so it needs an
// independently guarded helper.
$$ if HAS_HEIGHT_MAP is defined
fn getHeightTangentFrame(
    eye_pos: vec3<f32>,
    surf_norm: vec3<f32>,
    uv: vec2<f32>,
) -> mat3x3<f32> {
    let q0 = dpdx(eye_pos);
    let q1 = dpdy(eye_pos);
    let st0 = dpdx(uv);
    let st1 = dpdy(uv);
    let q1perp = cross(q1, surf_norm);
    let q0perp = cross(surf_norm, q0);
    let tangent = q1perp * st0.x + q0perp * st1.x;
    let bitangent = q1perp * st0.y + q0perp * st1.y;
    let determinant = max(dot(tangent, tangent), dot(bitangent, bitangent));
    let scale = select(inverseSqrt(determinant), 0.0, determinant == 0.0);

    return mat3x3f(tangent * scale, -bitangent * scale, surf_norm);
}
$$ endif

// ── Screen / Transient BindGroup (Group 3) ──────────────────────────
//
// Bindings 0-7 are always present. Clustered variants append bindings 8-10
// so non-clustered pipelines keep the original forward layout.
@group(3) @binding(1) var s_screen_sampler: sampler;
@group(3) @binding(2) var t_ssao: texture_2d<f32>;
@group(3) @binding(3) var t_shadow_map_2d_array: texture_depth_2d_array;
@group(3) @binding(4) var t_shadow_map_cube_array: texture_depth_cube_array;
@group(3) @binding(5) var s_shadow_map_compare: sampler_comparison;
@group(3) @binding(6) var<uniform> u_local_light_buffer_metadata: LightBufferMetadata;
@group(3) @binding(7) var<storage, read> st_local_lights: array<Struct_lights>;
$$ if USE_CLUSTERED_SHADING is defined
@group(3) @binding(8) var<uniform> u_clustered_lighting: ClusteredLightingParams;
@group(3) @binding(9) var<storage, read> st_cluster_records: array<ClusterRecord>;
@group(3) @binding(10) var<storage, read> st_cluster_light_indices: array<u32>;
$$ endif

@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var local_position = vec3<f32>(in.position.xyz);

    $$ if HAS_NORMAL is defined
    var local_normal = vec3<f32>(in.normal.xyz);
    $$ endif

    $$ if HAS_TANGENT is defined
    var object_tangent = vec3<f32>(in.tangent.xyz);
    $$ endif

    // ── Morph Target Blending ────────────────────────────────────────
    $$ if HAS_MORPH_TARGETS
        let morphed = apply_morph_targets(
            vertex_index,
            local_position,
            $$ if HAS_MORPH_NORMALS and HAS_NORMAL
            local_normal,
            $$ endif
            $$ if HAS_MORPH_TANGENTS and HAS_TANGENT
            object_tangent,
            $$ endif
        );
        local_position = morphed.position;
        $$ if HAS_MORPH_NORMALS and HAS_NORMAL
        local_normal = morphed.normal;
        $$ endif
        $$ if HAS_MORPH_TANGENTS and HAS_TANGENT
        object_tangent = morphed.tangent;
        $$ endif
    $$ endif

    var local_pos = vec4<f32>(local_position, 1.0);

    // ── Skeletal Skinning ────────────────────────────────────────────
    $$ if HAS_SKINNING and SUPPORT_SKINNING
        let skinned = compute_skinned_vertex(
            local_pos,
            $$ if HAS_NORMAL
            local_normal,
            $$ endif
            $$ if HAS_TANGENT
            object_tangent,
            $$ endif
            vec4<u32>(in.joints),
            in.weights,
        );
        local_pos = skinned.position;
        $$ if HAS_NORMAL
        local_normal = skinned.normal;
        $$ endif
        $$ if HAS_TANGENT
        object_tangent = skinned.tangent;
        $$ endif
    $$ endif

    let world_pos = u_model.world_matrix * local_pos;

    $$ if IN_TRANSPARENT_PASS
        out.position = u_render_state.unjittered_view_projection * world_pos;
    $$ else
        out.position = u_render_state.view_projection * world_pos;
    $$ endif

    out.world_position = world_pos.xyz / world_pos.w;

    $$ if HAS_COLOR
    out.color = in.color;
    $$ endif

    $$ if HAS_UV
    out.uv = in.uv;
    $$ endif

    $$ if HAS_NORMAL
    out.geometry_normal = local_normal;
    out.normal = normalize(u_model.normal_matrix * local_normal);
    $$ endif

    $$ if HAS_TANGENT
    var v_tangent = normalize(( u_model.world_matrix  * vec4f(object_tangent, 0.0) ).xyz);
    v_tangent = normalize(v_tangent - out.normal * dot(out.normal, v_tangent));
    let v_bitangent = normalize(cross(out.normal, v_tangent) * in.tangent.w);
    out.v_tangent = vec3<f32>(v_tangent);
    out.v_bitangent = vec3<f32>(v_bitangent);
    $$ endif

    {$ include 'mixins/uv_vertex' $}
    return out;
}


@fragment
fn fs_main(
    varyings: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {

    let face_direction = f32(is_front) * 2.0 - 1.0;

    $$ if FLAT_SHADING or HAS_NORMAL is not defined
        let u = dpdx(varyings.world_position);
        let v = dpdy(varyings.world_position);
        var surface_normal = normalize(cross(u, v));
    $$ else
        var surface_normal = normalize(vec3<f32>(varyings.normal));
        surface_normal = surface_normal * face_direction;
    $$ endif

    var diffuse_color = u_material.color;

    $$ if HAS_COLOR
        diffuse_color *= varyings.color;
    $$ endif

    $$ if HAS_MAP
        let tex_color = textureSample(t_map, s_map, varyings.map_uv);
        diffuse_color *= tex_color;
        if (u_material.overlay.a > 0.0) {
            let overlay_luma = dot(tex_color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            diffuse_color = vec4<f32>(
                u_material.overlay.rgb * (overlay_luma / u_material.overlay.a),
                diffuse_color.a,
            );
        }
    $$ endif

    // Apply opacity
    var opacity = diffuse_color.a * u_material.opacity;

    // Alpha test
    $$ if ALPHA_MODE == "MASK" or ALPHA_MODE == "BLEND_MASK"
    apply_alpha_test(&opacity, u_material.alpha_test);
    $$ endif

    let view = normalize(u_render_state.camera_position - varyings.world_position);

    $$ if HAS_NORMAL_MAP is defined or HAS_HEIGHT_MAP is defined or USE_ANISOTROPY is defined
        $$ if HAS_TANGENT is defined
            var tbn = mat3x3f(normalize(varyings.v_tangent), normalize(varyings.v_bitangent), surface_normal);
        $$ else
            $$ if HAS_NORMAL_MAP is defined
                let n_uv = varyings.normal_map_uv; 
            $$ elif HAS_HEIGHT_MAP is defined
                let n_uv = varyings.height_map_uv;
            $$ elif HAS_CLEARCOAT_NORMAL_MAP is defined
                let n_uv = varyings.clearcoat_normal_map_uv;
            $$ elif HAS_MAP_UV is defined
                let n_uv = varyings.map_uv;
            $$ else
                let n_uv = varyings.uv;
            $$ endif
            $$ if HAS_NORMAL_MAP is defined or HAS_CLEARCOAT_NORMAL_MAP is defined or USE_ANISOTROPY is defined
                var tbn = getTangentFrame(view, surface_normal, n_uv);
            $$ else
                var tbn = getHeightTangentFrame(view, surface_normal, n_uv);
            $$ endif
        $$ endif

        tbn[0] = tbn[0] * face_direction;
        tbn[1] = tbn[1] * face_direction;
    $$ endif

    $$ if HAS_NORMAL_MAP is defined
        let normal_map = textureSample( t_normal_map, s_normal_map, varyings.normal_map_uv ) * 2.0 - 1.0;
        let map_n = vec3f(normal_map.xy * u_material.normal_scale, normal_map.z);
        var normal = normalize(tbn * map_n);
    $$ else
        var normal = surface_normal;
    $$ endif

    // Height maps contribute a tangent-space slope. This changes the light's
    // reading of shallow relief without moving the mesh silhouette.
    $$ if HAS_HEIGHT_MAP is defined
        let height_dimensions = vec2<f32>(textureDimensions(t_height_map));
        let height_texel = 1.0 / height_dimensions;
        let height_here = textureSample(
            t_height_map,
            s_height_map,
            varyings.height_map_uv,
        ).r;
        let height_u = textureSample(
            t_height_map,
            s_height_map,
            varyings.height_map_uv + vec2<f32>(height_texel.x, 0.0),
        ).r;
        let height_v = textureSample(
            t_height_map,
            s_height_map,
            varyings.height_map_uv + vec2<f32>(0.0, height_texel.y),
        ).r;
        let height_slope = vec2<f32>(
            height_here - height_u,
            height_here - height_v,
        ) * height_dimensions * u_material.height_scale;
        normal = normalize(normal + tbn * vec3<f32>(height_slope, 0.0));
    $$ endif

    $$ if USE_CLEARCOAT is defined
        $$ if HAS_CLEARCOAT_NORMAL_MAP is defined
            $$ if HAS_TANGENT is defined
                var tbn_cc = mat3x3f(varyings.v_tangent, varyings.v_bitangent, surface_normal);
            $$ else
                var tbn_cc = getTangentFrame( view, surface_normal, varyings.clearcoat_normal_map_uv );
            $$ endif

            tbn_cc[0] = tbn_cc[0] * face_direction;
            tbn_cc[1] = tbn_cc[1] * face_direction;

            var clearcoat_normal_map = textureSample(t_clearcoat_normal_map, s_clearcoat_normal_map, varyings.clearcoat_normal_map_uv ) * 2.0 - 1.0;
            let clearcoat_map_n = vec3f(clearcoat_normal_map.xy * u_material.clearcoat_normal_scale, clearcoat_normal_map.z);
            let clearcoat_normal = normalize(tbn_cc * clearcoat_map_n);
        $$ else
            let clearcoat_normal = surface_normal;
        $$ endif
    $$ endif

    $$ if HAS_SPECULAR_MAP is defined
        let specular_map = textureSample( t_specular_map, s_specular_map, varyings.specular_map_uv );
        let specular_strength = specular_map.r;
    $$ else
        let specular_strength = 1.0;
    $$ endif

    var reflected_light: ReflectedLight = ReflectedLight(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));

    var geometry: GeometricContext;
    geometry.position = varyings.world_position;
    geometry.normal = normal;
    geometry.view_dir = view;

    $$ if USE_CLEARCOAT is defined
        geometry.clearcoat_normal = clearcoat_normal;
    $$ endif

    {$ include 'mixins/physical_material_setup' $}

    // Uniform roughness establishes the base response. The strength controls
    // how much authored per-texel variation is mixed into it.
    $$ if HAS_ROUGHNESS_MAP is defined
        let authored_roughness = textureSample(
            t_roughness_map,
            s_roughness_map,
            varyings.roughness_map_uv,
        ).g;
        roughness_factor = u_material.roughness * mix(
            1.0,
            authored_roughness,
            u_material.roughness_map_strength,
        );
        material.roughness = clamp(
            max(roughness_factor, 0.0525) + geometry_roughness,
            0.0,
            1.0,
        );
    $$ endif

    // ── Debug View: material attribute short-circuit ──────────────────
    $$ if DEBUG_VIEW_ALBEDO is defined
        return pack_fragment_output(vec4<f32>(diffuse_color.rgb, 1.0));
    $$ endif
    $$ if DEBUG_VIEW_ROUGHNESS is defined
        return pack_fragment_output(vec4<f32>(vec3<f32>(roughness_factor), 1.0));
    $$ endif
    $$ if DEBUG_VIEW_METALNESS is defined
        return pack_fragment_output(vec4<f32>(vec3<f32>(metalness_factor), 1.0));
    $$ endif

    evaluate_punctual_lights(geometry, material, &reflected_light, varyings.position);

    // Indirect Diffuse Light
    let ambient_color = u_environment.ambient_light.rgb;
    var irradiance = getAmbientLightIrradiance( ambient_color );

    $$ if HAS_LIGHT_MAP is defined
        let light_map_color = textureSample(t_light_map, s_light_map, varyings.light_map_uv ).rgb;
        irradiance += light_map_color * u_material.light_map_intensity;
    $$ endif

    RE_IndirectDiffuse( irradiance, geometry, material, &reflected_light );

    $$ if USE_IBL is defined

        let s = sin(u_environment.env_map_rotation);
        let c = cos(u_environment.env_map_rotation);

        let ibl_rotated_view = vec3<f32>(
            view.x * c - view.z * s,
            view.y,
            view.x * s + view.z * c
        );

        $$ if USE_ANISOTROPY is defined
            let ibl_radiance = getIBLAnisotropyRadiance( ibl_rotated_view, normal, material.roughness, material.anisotropy_b, material.anisotropy );
        $$ else
            let ibl_radiance = getIBLRadiance( ibl_rotated_view, normal, material.roughness);
        $$ endif

        var clearcoat_ibl_radiance = vec3<f32>(0.0);
        $$ if USE_CLEARCOAT is defined
            clearcoat_ibl_radiance += getIBLRadiance( ibl_rotated_view, clearcoat_normal, material.clearcoat_roughness );
        $$ endif

        let ibl_rotated_normal = vec3<f32>(
            normal.x * c - normal.z * s,
            normal.y,
            normal.x * s + normal.z * c
        );
        let ibl_irradiance = getIBLIrradiance( ibl_rotated_normal );
        RE_IndirectSpecular(ibl_radiance, ibl_irradiance, clearcoat_ibl_radiance, geometry, material, &reflected_light);
    $$ endif

    // ── Ambient Occlusion ────────────────────────────────────────────

    var ambient_occlusion = 1.0;
    $$ if HDR and USE_SSAO
    let screen_ndc = varyings.position.xy / varyings.position.w;
    let screen_uv = vec2<f32>(
        screen_ndc.x * 0.5 + 0.5,
        screen_ndc.y * -0.5 + 0.5
    );
    ambient_occlusion = textureSampleLevel(t_ssao, s_screen_sampler, screen_uv, 0.0).r;
    $$ endif

    $$ if HAS_AO_MAP is defined
        let ao_map_intensity = u_material.ao_map_intensity;
        let material_ao = ( textureSample( t_ao_map, s_ao_map, varyings.ao_map_uv ).r - 1.0 ) * ao_map_intensity + 1.0;
        ambient_occlusion *= material_ao;
    $$ endif

    reflected_light.indirect_diffuse *= ambient_occlusion;

    $$ if USE_CLEARCOAT is defined
        clearcoat_specular_indirect *= ambient_occlusion;
    $$ endif

    $$ if USE_SHEEN is defined
        sheen_specular_indirect *= ambient_occlusion;
    $$ endif

    $$ if USE_IBL is defined
        let dot_nv = saturate( dot( geometry.normal, geometry.view_dir ) );
        reflected_light.indirect_specular *= computeSpecularOcclusion( dot_nv, ambient_occlusion, material.roughness );
    $$ endif


    var total_diffuse = reflected_light.direct_diffuse + reflected_light.indirect_diffuse;
    var total_specular = reflected_light.direct_specular + reflected_light.indirect_specular;

    // ── Volume Transmission ──────────────────────────────────────────
    $$ if USE_TRANSMISSION is defined
        let pos = varyings.world_position;
        let v = normalize(u_render_state.camera_position - pos);
        let n = surface_normal;
        let model_matrix = u_model.world_matrix;

        $$ if IN_TRANSPARENT_PASS
            let view_projection_matrix = u_render_state.unjittered_view_projection;
        $$ else
            let view_projection_matrix = u_render_state.view_projection;
        $$ endif

        let transmitted = getIBLVolumeRefraction(
            n, v, material.roughness, material.diffuse_color, material.specular_color, material.specular_f90,
            pos, model_matrix, view_projection_matrix, material.dispersion, material.ior, material.thickness,
            material.attenuation_color, material.attenuation_distance );

        material.transmission_alpha = mix( material.transmission_alpha, transmitted.a, material.transmission );

        total_diffuse = mix( total_diffuse, transmitted.rgb, material.transmission );
    $$ endif

    // ── Final Compositing ────────────────────────────────────────────

    var out_diffuse = total_diffuse;
    var out_specular = total_specular;

    // Emissive
    var emissive_color = u_material.emissive.rgb * u_material.emissive_intensity;
    $$ if HAS_EMISSIVE_MAP is defined
        emissive_color *= textureSample(t_emissive_map, s_emissive_map, varyings.emissive_map_uv).rgb;
    $$ endif
    out_diffuse += emissive_color;

    // Sheen energy compensation
    $$ if USE_SHEEN is defined
        let sheen_energy_comp = 1.0 - 0.157 * max(material.sheen_color.r, max(material.sheen_color.g, material.sheen_color.b));
        out_diffuse *= sheen_energy_comp;
        out_specular *= sheen_energy_comp;
        out_specular += (sheen_specular_direct + sheen_specular_indirect);
    $$ endif

    // Clearcoat energy attenuation
    $$ if USE_CLEARCOAT is defined
        let dot_nv_cc = saturate(dot(clearcoat_normal, view));
        let fcc = F_Schlick( material.clearcoat_f0, material.clearcoat_f90, dot_nv_cc );
        let clearcoat_attenuation = 1.0 - material.clearcoat * fcc;
        out_diffuse *= clearcoat_attenuation;
        out_specular *= clearcoat_attenuation;
        out_specular += (clearcoat_specular_direct + clearcoat_specular_indirect) * material.clearcoat;
    $$ endif

    $$ if OPAQUE is defined
        opacity = 1.0;
    $$ endif

    $$ if USE_TRANSMISSION is defined
        opacity *= material.transmission_alpha;
    $$ endif

    // Output
    var out: FragmentOutput;

    $$ if HDR
    out_diffuse = clamp(out_diffuse, vec3<f32>(0.0), vec3<f32>(65000.0));
    out_specular = clamp(out_specular, vec3<f32>(0.0), vec3<f32>(65000.0));
    $$ endif

    var out_color = out_diffuse + out_specular;
    $$ if HAS_MRT_SPECULAR_DATA is defined and USE_SSS is defined
        if (u_material.sss_id != 0u) {
            out_color = out_diffuse;
        }
    $$ endif
    $$ if not HDR
        out_color = apply_pbr_tone_mapping(out_color);
    $$ endif
    out.color = vec4<f32>(out_color, opacity);

    $$ if HAS_MRT_SPECULAR_DATA is defined
    out.specular_data = vec4<f32>(out_specular, metalness_factor);
    $$ endif

    $$ if HAS_MRT_MATERIAL_DATA is defined
    out.material_data = vec4<f32>(material.diffuse_color, material.roughness);
    $$ endif

    return out;
}
