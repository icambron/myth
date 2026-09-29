//! RDG Shadow Render Pass
//!
//! Renders shadow depth maps for all shadow-casting lights. Three transient
//! textures are managed per frame:
//!
//! - **`Texture2DArray`** — directional light cascades.
//! - **`Texture2DArray`** — spot light shadows, at an independent resolution.
//! - **`TextureCubeArray`** — point light omnidirectional shadows (6 faces per
//!   light, hardware-seamless PCF across edges).
//!
//! VP matrices are uploaded to a per-layer dynamic uniform buffer shared by
//! all shadow textures.
//!
//! # RDG Integration
//!
//! All shadow map textures are **transient RDG resources** — their lifetimes
//! are fully managed by the render graph compiler. Consumer passes (Opaque,
//! Transparent, etc.) declare explicit `read_texture` dependencies to
//! establish DAG ordering and enable automatic memory aliasing.
//!
//! # Data Flow
//!
//! ```text
//! RenderLists::active_views (shadow views)     ← built by extract_and_prepare
//! RenderLists::shadow_queues (per-view cmds)   ← built by culling::cull_and_sort
//!         │
//!         ▼
//!  ┌──────────────────────────────────────┐
//!  │   ShadowPass                         │
//!  │  setup:   create_texture (RDG)       │
//!  │  prepare: resolve physical → views   │
//!  │  execute: draw depth per layer       │
//!  └──────────────────────────────────────┘
//!         │
//!         ▼  (TextureNodeId via explicit wiring)
//!  Consumer passes (Opaque / Transparent)
//!         read_texture(shadow_2d_id) → DAG edge
//!         read_texture(shadow_cube_id) → DAG edge
//! ```

use glam::Mat4;
use myth_scene::light::LightKind;

use crate::core::gpu::Tracked;
use crate::core::view::ViewTarget;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{ExecuteContext, ExtractContext, PassNode, TextureDesc, TextureNodeId};
use crate::graph::frame::ShadowLightInstance;
use crate::graph::passes::draw::submit_draw_commands;

/// Output returned by [`ShadowFeature::add_to_graph`].
///
/// Consumer passes wire `read_texture` dependencies against these IDs.
#[derive(Clone, Copy, Default)]
pub struct ShadowOutput {
    /// 2D array shadow map (directional). `None` when unused.
    pub shadow_2d: Option<TextureNodeId>,
    /// Cube array shadow map (point lights). `None` when unused.
    pub shadow_cube: Option<TextureNodeId>,
    /// Independent-resolution 2D array for spot lights.
    pub shadow_spot: Option<TextureNodeId>,
}

/// Shadow rendering feature.
///
/// Manages a dynamic uniform buffer for per-layer VP matrices and produces
/// ephemeral [`ShadowPassNode`] instances each frame via [`Self::add_to_graph`].
///
/// The shadow map textures are **transient RDG resources** — their physical
/// memory is allocated by the graph compiler and can be aliased with other
/// non-overlapping intermediates. If no shadow-casting lights are active,
/// no GPU memory is allocated at all.
pub struct ShadowFeature {
    /// Per-light VP matrix buffer (dynamic uniform, stride-aligned).
    uniform_buffer: Tracked<wgpu::Buffer>,
    /// Current buffer capacity in layers.
    uniform_capacity: u32,
    /// Aligned stride between consecutive VP matrices.
    uniform_stride: u32,
    /// Bind group referencing `uniform_buffer`.
    bind_group: Option<wgpu::BindGroup>,
    /// Bind group layout for shadow light uniforms.
    bind_group_layout: Tracked<wgpu::BindGroupLayout>,

    /// All shadow light instances this frame (both 2D and cube).
    shadow_lights: Vec<ShadowLightInstance>,
    /// Pre-built (uniform_index, instance) pairs for directional shadows.
    d2_lights: Vec<(u32, ShadowLightInstance)>,
    /// Pre-built (uniform_index, instance) pairs for cube shadows.
    cube_lights: Vec<(u32, ShadowLightInstance)>,
    /// Total cube array layers (point lights, always a multiple of 6).
    total_cube_layers: u32,
    /// Maximum shadow map resolution across directional views.
    shadow_2d_resolution: u32,
    /// Pre-built (uniform_index, instance) pairs for spot shadows.
    spot_lights: Vec<(u32, ShadowLightInstance)>,
    /// Maximum shadow map resolution across spot views.
    shadow_spot_resolution: u32,
    /// Maximum shadow map resolution across cube views.
    shadow_cube_resolution: u32,
}

impl ShadowFeature {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let min_alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let stride = align_to(std::mem::size_of::<Mat4>() as u32, min_alignment);

        let bind_group_layout = Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow Light BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Mat4>() as u64),
                    },
                    count: None,
                }],
            },
        ));

        let uniform_buffer = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow VP Buffer"),
            size: u64::from(stride),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        Self {
            uniform_buffer,
            uniform_capacity: 1,
            uniform_stride: stride,
            bind_group: None,
            bind_group_layout,
            shadow_lights: Vec::with_capacity(16),
            d2_lights: Vec::with_capacity(16),
            cube_lights: Vec::with_capacity(16),
            total_cube_layers: 0,
            shadow_2d_resolution: 1,
            spot_lights: Vec::new(),
            shadow_spot_resolution: 1,
            shadow_cube_resolution: 1,
        }
    }

    /// Grow the uniform buffer (and recreate the bind group) if the current
    /// capacity is insufficient.
    fn ensure_capacity(&mut self, device: &wgpu::Device, required_count: u32) -> bool {
        if required_count <= self.uniform_capacity {
            return false;
        }

        let mut capacity = self.uniform_capacity.max(1);
        while capacity < required_count {
            capacity = capacity.saturating_mul(2);
        }

        self.uniform_buffer = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow VP Buffer"),
            size: u64::from(self.uniform_stride) * u64::from(capacity),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.uniform_capacity = capacity;
        true
    }

    fn rebuild_bind_group(&mut self, ctx: &mut ExtractContext) {
        self.bind_group = Some(
            ctx.build_bind_group(&self.bind_group_layout, Some("Shadow Light BG"))
                .bind_tracked_buffer_with_size(
                    0,
                    &self.uniform_buffer,
                    wgpu::BufferSize::new(std::mem::size_of::<Mat4>() as u64),
                )
                .build()
                .clone(),
        );
    }
}

impl ShadowFeature {
    /// Extract shadow light data and prepare GPU resources.
    ///
    /// Classifies shadow views into 2D (directional/spot) and cube (point)
    /// categories, uploads per-layer VP matrices, and builds the shadow light
    /// instance list. Physical shadow textures are allocated by the RDG
    /// transient pool after graph compilation.
    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        self.shadow_lights.clear();
        self.d2_lights.clear();
        self.spot_lights.clear();
        self.shadow_spot_resolution = 1;
        self.cube_lights.clear();
        self.total_cube_layers = 0;
        self.shadow_2d_resolution = 1;
        self.shadow_cube_resolution = 1;

        let light_kinds: rustc_hash::FxHashMap<_, _> = ctx
            .extracted_scene
            .lights
            .iter()
            .map(|light| (light.id, &light.kind))
            .collect();

        // Count layers by type
        let total_shadow_views: u32 = ctx
            .render_lists
            .active_views
            .iter()
            .filter(|v| v.is_shadow())
            .count() as u32;

        if total_shadow_views == 0 {
            return;
        }

        // Classify views: assign separate layer indices for 2D and cube arrays.
        // The uniform buffer holds ALL VP matrices contiguously (indexed by a
        // global uniform_index). The layer_index within each texture type
        // is tracked separately.

        let mut uniform_index = 0u32;
        let mut d2_layer = 0u32;
        let mut cube_layer = 0u32;

        for view in &ctx.render_lists.active_views {
            let ViewTarget::ShadowLight {
                light_id,
                layer_index: _,
            } = view.target
            else {
                continue;
            };

            let is_point = matches!(light_kinds.get(&light_id), Some(LightKind::Point(_)));
            let is_spot = matches!(light_kinds.get(&light_id), Some(LightKind::Spot(_)));

            if is_point {
                self.shadow_cube_resolution = self.shadow_cube_resolution.max(view.viewport_size.0);

                let view_layer_index = match view.target {
                    ViewTarget::ShadowLight { layer_index, .. } => layer_index,
                    ViewTarget::MainCamera => 0,
                };

                let inst = ShadowLightInstance {
                    light_id,
                    view_layer_index,
                    texture_layer_index: cube_layer,
                    light_buffer_index: view.light_buffer_index,
                    light_view_projection: view.view_projection,
                    is_point: true,
                };
                self.shadow_lights.push(inst);
                self.cube_lights.push((uniform_index, inst));
                cube_layer += 1;
            } else if is_spot {
                self.shadow_spot_resolution = self.shadow_spot_resolution.max(view.viewport_size.0);
                let ViewTarget::ShadowLight {
                    layer_index: view_layer_index,
                    ..
                } = view.target
                else {
                    unreachable!()
                };
                let inst = ShadowLightInstance {
                    light_id,
                    view_layer_index,
                    texture_layer_index: self.spot_lights.len() as u32,
                    light_buffer_index: view.light_buffer_index,
                    light_view_projection: view.view_projection,
                    is_point: false,
                };
                self.shadow_lights.push(inst);
                self.spot_lights.push((uniform_index, inst));
            } else {
                self.shadow_2d_resolution = self.shadow_2d_resolution.max(view.viewport_size.0);

                let view_layer_index = match view.target {
                    ViewTarget::ShadowLight { layer_index, .. } => layer_index,
                    ViewTarget::MainCamera => 0,
                };

                let inst = ShadowLightInstance {
                    light_id,
                    view_layer_index,
                    texture_layer_index: d2_layer,
                    light_buffer_index: view.light_buffer_index,
                    light_view_projection: view.view_projection,
                    is_point: false,
                };
                self.shadow_lights.push(inst);
                self.d2_lights.push((uniform_index, inst));
                d2_layer += 1;
            }

            uniform_index += 1;
        }

        self.total_cube_layers = cube_layer;

        // Ensure buffer capacity for all uniform slots
        let total_uniform_slots = uniform_index;
        let resized = self.ensure_capacity(ctx.device, total_uniform_slots);
        if resized || self.bind_group.is_none() {
            self.rebuild_bind_group(ctx);
        }

        // Upload VP matrices — one contiguous range, indexed by insertion order
        let mut uniform_data =
            vec![0u8; self.uniform_stride as usize * total_uniform_slots as usize];

        for (i, sl) in self.shadow_lights.iter().enumerate() {
            let offset = i * self.uniform_stride as usize;
            let bytes = bytemuck::bytes_of(&sl.light_view_projection);
            uniform_data[offset..offset + bytes.len()].copy_from_slice(bytes);
        }

        ctx.queue
            .write_buffer(&self.uniform_buffer, 0, &uniform_data);
    }

    /// Create ephemeral shadow pass nodes and add them to the render graph.
    ///
    /// Registers up to three transient `Depth32Float` textures:
    /// - Independent **2D arrays** for directional and spot shadow layers.
    /// - A **cube array** for point light shadow faces.
    ///
    /// Returns [`ShadowOutput`] containing the [`TextureNodeId`]s so that
    /// consumer passes can wire explicit read dependencies.
    pub fn add_to_graph<'a>(&'a self, ctx: &mut GraphBuilderContext<'a, '_>) -> ShadowOutput {
        let mut output = ShadowOutput {
            shadow_2d: None,
            shadow_cube: None,
            shadow_spot: None,
        };

        if self.shadow_lights.is_empty() {
            return output;
        }

        let bind_group = self
            .bind_group
            .as_ref()
            .expect("Shadow bind group must be prepared before graph build");

        // Collect 2D and cube shadow light slices
        let has_cube = self.total_cube_layers > 0;

        for (resolution, lights, output_slot, label) in [
            (
                self.shadow_2d_resolution,
                self.d2_lights.as_slice(),
                &mut output.shadow_2d,
                "Shadow_Directional_Array",
            ),
            (
                self.shadow_spot_resolution,
                self.spot_lights.as_slice(),
                &mut output.shadow_spot,
                "Shadow_Spot_Array",
            ),
        ] {
            if lights.is_empty() {
                continue;
            }
            let desc = TextureDesc::new(
                resolution,
                resolution,
                lights.len() as u32,
                1,
                1,
                wgpu::TextureDimension::D2,
                wgpu::TextureFormat::Depth32Float,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            let stride = self.uniform_stride;
            *output_slot = Some(ctx.graph.add_pass(label, |builder| {
                let tex_id = builder.create_texture(label, desc);
                let node = Shadow2DPassNode {
                    bind_group,
                    lights,
                    uniform_stride: stride,
                    texture_id: tex_id,
                };
                (node, tex_id)
            }));
        }

        if has_cube {
            let desc = TextureDesc::new(
                self.shadow_cube_resolution,
                self.shadow_cube_resolution,
                self.total_cube_layers, // must be multiple of 6
                1,
                1,
                wgpu::TextureDimension::D2,
                wgpu::TextureFormat::Depth32Float,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let cube_lights = self.cube_lights.as_slice();
            let stride = self.uniform_stride;
            let shadow_cube_id = ctx.graph.add_pass("Shadow_Cube_Pass", |builder| {
                let tex_id = builder.create_texture("Shadow_Cube_Array", desc);

                let node = ShadowCubePassNode {
                    bind_group,
                    lights: cube_lights,
                    uniform_stride: stride,
                    texture_id: tex_id,
                };
                (node, tex_id)
            });

            output.shadow_cube = Some(shadow_cube_id);
        }

        output
    }
}

// ─── 2D Array Shadow Pass Node ────────────────────────────────────────────────

/// Renders depth layers for directional and spot light shadows into a 2D
/// array texture.
struct Shadow2DPassNode<'a> {
    bind_group: &'a wgpu::BindGroup,
    /// (uniform_index, instance) pairs for 2D shadows.
    lights: &'a [(u32, ShadowLightInstance)],
    uniform_stride: u32,
    texture_id: TextureNodeId,
}

impl<'a> PassNode<'a> for Shadow2DPassNode<'a> {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if self.lights.is_empty() {
            return;
        }

        let texture = ctx.get_texture(self.texture_id);

        for &(uniform_idx, shadow_light) in self.lights {
            let layer_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Shadow_2D_Layer"),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: shadow_light.texture_layer_index,
                array_layer_count: Some(1),
                ..Default::default()
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Depth (2D)"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &layer_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let dynamic_offset = uniform_idx * self.uniform_stride;
            pass.set_bind_group(0, self.bind_group, &[dynamic_offset]);

            if let Some(commands) = ctx
                .baked_lists
                .shadow_queues
                .get(&(shadow_light.light_id, shadow_light.view_layer_index))
            {
                submit_draw_commands(&mut pass, commands);
            }
        }
    }
}

// ─── Cube Array Shadow Pass Node ──────────────────────────────────────────────

/// Renders depth layers for point light shadows into a cube array texture.
///
/// Each point light occupies 6 consecutive layers (one per cube face).
/// The texture is created as a 2D array with `depth_or_array_layers = 6 * N`,
/// and consumers view it as `TextureViewDimension::CubeArray` for seamless
/// hardware-filtered PCF sampling.
struct ShadowCubePassNode<'a> {
    bind_group: &'a wgpu::BindGroup,
    /// (uniform_index, instance) pairs for cube shadows.
    lights: &'a [(u32, ShadowLightInstance)],
    uniform_stride: u32,
    texture_id: TextureNodeId,
}

impl<'a> PassNode<'a> for ShadowCubePassNode<'a> {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if self.lights.is_empty() {
            return;
        }

        let texture = ctx.get_texture(self.texture_id);

        for &(uniform_idx, shadow_light) in self.lights {
            // Each face is a single 2D layer within the cube array texture.
            let layer_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Shadow_Cube_Face"),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: shadow_light.texture_layer_index,
                array_layer_count: Some(1),
                ..Default::default()
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Depth (Cube)"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &layer_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let dynamic_offset = uniform_idx * self.uniform_stride;
            pass.set_bind_group(0, self.bind_group, &[dynamic_offset]);

            if let Some(commands) = ctx
                .baked_lists
                .shadow_queues
                .get(&(shadow_light.light_id, shadow_light.view_layer_index))
            {
                submit_draw_commands(&mut pass, commands);
            }
        }
    }
}

#[inline]
fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}
