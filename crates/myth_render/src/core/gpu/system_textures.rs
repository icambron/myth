//! System Textures
//!
//! Global pool of 1×1 fallback textures used as default bind-group fillers
//! when optional rendering features (SSAO, shadows, transmission, etc.) are
//! disabled.  Textures are classified by **data semantics** rather than by
//! the feature they serve, eliminating per-feature dummy proliferation.
//!
//! # Design Rationale
//!
//! GPU bind-group layouts are strict: every declared slot must be bound even
//! when the feature that writes it is inactive.  Instead of creating one
//! dedicated dummy per feature (which couples low-level resource management
//! to high-level pipeline knowledge), `SystemTextures` provides a small set
//! of mathematically-neutral textures:
//!
//! | Texture          | Value              | Typical Usage                   |
//! |------------------|--------------------|----------------------------------|
//! | `white_2d`       | `[255,255,255,255]`| Multiplicative identity (AO, shadow) |
//! | `black_2d`       | `[0,0,0,255]`      | Additive identity (emission)     |
//! | `transparent_2d` | `[0,0,0,0]`        | No contribution (transmission)   |
//! | `normal_2d`      | `[128,128,255,255]`| Tangent-space +Z (flat normal)   |
//! | `black_cube`     | 6×`[0,0,0,255]`    | Empty environment / IBL          |
//! | `white_r8`       | `[255]` R8Unorm    | SSAO fallback (fully lit)        |
//! | `black_hdr`      | Rgba16Float zero   | Transmission HDR fallback        |
//! | `depth_d2array`  | Depth32Float 1×1   | Shadow map D2Array fallback      |
//! | `blue_noise`     | 64×64 RGBA8        | Stochastic sampling source       |
//!
//! # Screen Bind Group (Group 3)
//!
//! The struct also owns the **layout**, **samplers**, and
//! **`build_bind_group`** logic for the screen-level descriptor set
//! (Group 3), consolidating everything previously scattered across
//! `ScreenBindGroupInfo` and `ResourceManager`.

use super::Tracked;
use myth_resources::uniforms::{
    ClusterRecord, ClusteredLightingParams, GpuLightStorage, LightBufferMetadata,
};

const BLUE_NOISE_SIZE: u32 = 64;

#[cfg(feature = "advanced_noise")]
const BLUE_NOISE_LAYER_COUNT: u32 = 64;

#[cfg(not(feature = "advanced_noise"))]
const BLUE_NOISE_LAYER_COUNT: u32 = 1;

#[cfg(feature = "advanced_noise")]
const BLUE_NOISE_BYTES: &[u8] =
    include_bytes!("../../../assets/internal/blue_noise_array_64_rgba.png");

#[cfg(not(feature = "advanced_noise"))]
const BLUE_NOISE_BYTES: &[u8] =
    include_bytes!("../../../assets/internal/blue_noise_single_64_rgba.png");

/// Global system fallback texture pool and Group 3 bind-group infrastructure.
///
/// Created once during renderer initialisation and shared (by reference)
/// with all render-graph contexts. Fallback textures are 1×1 (or 1×1×6 for
/// cubes); the only larger resident assets are the feature-gated packed blue-noise PNGs.
pub struct SystemTextures {
    // ─── Data-Semantic Fallback Textures ───────────────────────────
    /// 1×1 RGBA8 `[255,255,255,255]` — multiplicative identity.
    pub white_2d: Tracked<wgpu::TextureView>,

    /// 1×1 RGBA8 `[0,0,0,255]` — additive identity.
    pub black_2d: Tracked<wgpu::TextureView>,

    /// 1×1 RGBA8 `[0,0,0,0]` — fully transparent / no contribution.
    pub transparent_2d: Tracked<wgpu::TextureView>,

    /// 1×1 RGBA8 `[128,128,255,255]` — default tangent-space normal (+Z).
    pub normal_2d: Tracked<wgpu::TextureView>,

    /// 1×1×6 RGBA8 all-black cube map — empty environment / IBL.
    pub black_cube: Tracked<wgpu::TextureView>,

    /// 1×1 R8Unorm `[255]` — SSAO fallback (AO = 1.0, fully lit).
    pub white_r8: Tracked<wgpu::TextureView>,

    /// 1×1 Rgba16Float zero — HDR transmission fallback.
    pub black_hdr: Tracked<wgpu::TextureView>,

    /// Feature-gated system blue-noise texture used by stochastic passes.
    pub blue_noise: Tracked<wgpu::TextureView>,

    /// 1×1 Depth32Float D2Array — shadow map fallback.
    pub depth_d2array: Tracked<wgpu::TextureView>,

    /// 1×1×6 Depth32Float CubeArray — point shadow map fallback.
    pub depth_cube_array: Tracked<wgpu::TextureView>,

    /// Default clustered-lighting parameters used when the scene pass has no
    /// active cluster buffers wired.
    pub clustered_params: Tracked<wgpu::Buffer>,

    /// Default scene-light metadata used when no explicit light buffer is bound.
    pub light_metadata: Tracked<wgpu::Buffer>,

    /// Default scene-light storage buffer containing a single inert light.
    pub light_storage: Tracked<wgpu::Buffer>,

    /// Default atmospheric bake parameters used when no procedural sky is bound.
    pub atmosphere_bake_params: Tracked<wgpu::Buffer>,

    /// Default cluster record buffer containing a single empty record.
    pub clustered_records: Tracked<wgpu::Buffer>,

    /// Default clustered light-index buffer containing a single zero entry.
    pub clustered_light_indices: Tracked<wgpu::Buffer>,

    // ─── Screen BindGroup Infrastructure (Group 3) ─────────────────
    /// `BindGroupLayout` for the base Group 3 screen-space resources.
    pub screen_layout: Tracked<wgpu::BindGroupLayout>,

    /// `BindGroupLayout` for clustered Group 3 pipelines.
    pub screen_layout_clustered: Tracked<wgpu::BindGroupLayout>,

    /// Linear-clamp sampler shared by transmission / SSAO sampling.
    pub screen_sampler: Tracked<wgpu::Sampler>,

    /// `LessEqual` comparison sampler for PCF shadow sampling.
    pub shadow_compare_sampler: Tracked<wgpu::Sampler>,

    /// Repeat + nearest sampler for blue-noise lookups.
    pub blue_noise_sampler: Tracked<wgpu::Sampler>,
}

impl SystemTextures {
    /// Creates all system textures and Group 3 infrastructure.
    ///
    /// This is called once during renderer initialisation. The GPU footprint is
    /// dominated by the optional blue-noise texture set; all fallback textures
    /// remain tiny.
    #[must_use]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // ── Data-Semantic Textures ─────────────────────────────────────

        let white_2d = create_1x1_rgba8(device, queue, [255, 255, 255, 255], "sys_white_2d");
        let black_2d = create_1x1_rgba8(device, queue, [0, 0, 0, 255], "sys_black_2d");
        let transparent_2d = create_1x1_rgba8(device, queue, [0, 0, 0, 0], "sys_transparent_2d");
        let normal_2d = create_1x1_rgba8(device, queue, [128, 128, 255, 255], "sys_normal_2d");
        let black_cube = create_1x1_cube(device, queue, [0, 0, 0, 255], "sys_black_cube");
        let white_r8 = create_1x1_r8(device, queue, 255, "sys_white_r8");
        let black_hdr = create_1x1_hdr(device, "sys_black_hdr");
        let blue_noise = create_blue_noise_texture(device, queue);
        let depth_d2array = create_1x1_depth_d2array(device, "sys_depth_d2array");
        let depth_cube_array = create_1x1_depth_cube_array(device, "sys_depth_cube_array");
        let clustered_params = create_default_clustered_params(device, queue);
        let light_metadata = create_default_light_metadata(device, queue);
        let light_storage = create_default_light_storage(device, queue);
        let atmosphere_bake_params = create_default_atmosphere_bake_params(device, queue);
        let clustered_records = create_default_clustered_records(device, queue);
        let clustered_light_indices = create_default_clustered_light_indices(device, queue);

        // ── Group 3 Layout ─────────────────────────────────────────────

        let screen_layout = create_screen_bind_group_layout(device, false);
        let screen_layout_clustered = create_screen_bind_group_layout(device, true);

        // ── Samplers ───────────────────────────────────────────────────

        let screen_sampler = Tracked::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Screen Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        let shadow_compare_sampler =
            Tracked::new(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Shadow Comparison Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual),
                ..Default::default()
            }));

        let blue_noise_sampler = Tracked::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("System Blue Noise Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        }));

        Self {
            white_2d,
            black_2d,
            transparent_2d,
            normal_2d,
            black_cube,
            white_r8,
            black_hdr,
            blue_noise,
            depth_d2array,
            depth_cube_array,
            clustered_params,
            light_metadata,
            light_storage,
            atmosphere_bake_params,
            clustered_records,
            clustered_light_indices,
            screen_layout,
            screen_layout_clustered,
            screen_sampler,
            shadow_compare_sampler,
            blue_noise_sampler,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Private helpers — minimal 1×1 texture construction
// ═══════════════════════════════════════════════════════════════════════════

fn create_screen_bind_group_layout(
    device: &wgpu::Device,
    clustered: bool,
) -> Tracked<wgpu::BindGroupLayout> {
    let mut entries = Vec::with_capacity(if clustered { 15 } else { 12 });

    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 2,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 3,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 4,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::CubeArray,
            multisampled: false,
        },
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 5,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 6,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 7,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });

    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 15,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    });

    if clustered {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 10,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }

    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 11,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 12,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 13,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: blue_noise_view_dimension(),
            multisampled: false,
        },
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 14,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    });

    Tracked::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(if clustered {
                "Screen/Transient Clustered Layout (Group 3)"
            } else {
                "Screen/Transient Layout (Group 3)"
            }),
            entries: &entries,
        }),
    )
}

fn blue_noise_view_dimension() -> wgpu::TextureViewDimension {
    if cfg!(feature = "advanced_noise") {
        wgpu::TextureViewDimension::D2Array
    } else {
        wgpu::TextureViewDimension::D2
    }
}

fn create_blue_noise_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Tracked<wgpu::TextureView> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sys_blue_noise"),
        size: wgpu::Extent3d {
            width: BLUE_NOISE_SIZE,
            height: BLUE_NOISE_SIZE,
            depth_or_array_layers: BLUE_NOISE_LAYER_COUNT,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let rgba = decode_blue_noise_image(BLUE_NOISE_BYTES);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(BLUE_NOISE_SIZE * 4),
            rows_per_image: Some(BLUE_NOISE_SIZE),
        },
        wgpu::Extent3d {
            width: BLUE_NOISE_SIZE,
            height: BLUE_NOISE_SIZE,
            depth_or_array_layers: BLUE_NOISE_LAYER_COUNT,
        },
    );

    Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("sys_blue_noise_view"),
        dimension: Some(blue_noise_view_dimension()),
        ..Default::default()
    }))
}

fn decode_blue_noise_image(png_bytes: &[u8]) -> Vec<u8> {
    let rgba = image::load_from_memory(png_bytes)
        .unwrap_or_else(|error| panic!("failed to decode packed blue-noise image: {error}"))
        .to_rgba8();

    assert_eq!(
        rgba.width(),
        BLUE_NOISE_SIZE,
        "packed blue-noise image has invalid width"
    );
    assert_eq!(
        rgba.height(),
        BLUE_NOISE_SIZE * BLUE_NOISE_LAYER_COUNT,
        "packed blue-noise image has invalid height"
    );

    rgba.into_raw()
}

fn create_1x1_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color: [u8; 4],
    label: &str,
) -> Tracked<wgpu::TextureView> {
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &color,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        size,
    );
    Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor::default()))
}

fn create_1x1_cube(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color: [u8; 4],
    label: &str,
) -> Tracked<wgpu::TextureView> {
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 6,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // 1×1 pixel × 4 bytes × 6 faces
    let mut data = [0u8; 24];
    for face in 0..6 {
        data[face * 4..face * 4 + 4].copy_from_slice(&color);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        size,
    );
    Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    }))
}

fn create_1x1_r8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    value: u8,
    label: &str,
) -> Tracked<wgpu::TextureView> {
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[value],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(1),
            rows_per_image: Some(1),
        },
        size,
    );
    Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor::default()))
}

/// Rgba16Float 1×1 zero-initialised texture (GPU zero-fills by default).
fn create_1x1_hdr(device: &wgpu::Device, label: &str) -> Tracked<wgpu::TextureView> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor::default()))
}

fn create_default_clustered_params(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Tracked<wgpu::Buffer> {
    let buffer = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sys_clustered_params"),
        size: std::mem::size_of::<ClusteredLightingParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    let params = ClusteredLightingParams {
        screen_dimensions: glam::UVec4::new(1, 1, 1, 1),
        grid_dimensions: glam::UVec4::new(1, 1, 1, 1),
        budget: glam::UVec4::new(1, 1, 0, 0),
        depth_params: glam::Vec4::new(0.1, 1.0, 0.0, 0.0),
    };
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&params));
    buffer
}

fn create_default_light_metadata(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Tracked<wgpu::Buffer> {
    let buffer = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sys_light_metadata"),
        size: std::mem::size_of::<LightBufferMetadata>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    queue.write_buffer(
        &buffer,
        0,
        bytemuck::bytes_of(&LightBufferMetadata::default()),
    );
    buffer
}

fn create_default_light_storage(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Tracked<wgpu::Buffer> {
    let buffer = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sys_light_storage"),
        size: std::mem::size_of::<GpuLightStorage>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&GpuLightStorage::default()));
    buffer
}

fn create_default_atmosphere_bake_params(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Tracked<wgpu::Buffer> {
    let zero_words = [0.0_f32; 20];
    let buffer = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sys_atmosphere_bake_params"),
        size: std::mem::size_of_val(&zero_words) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&zero_words));
    buffer
}

fn create_default_clustered_records(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Tracked<wgpu::Buffer> {
    let buffer = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sys_clustered_records"),
        size: std::mem::size_of::<ClusterRecord>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&ClusterRecord::default()));
    buffer
}

fn create_default_clustered_light_indices(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Tracked<wgpu::Buffer> {
    let buffer = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sys_clustered_light_indices"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&0u32));
    buffer
}

/// Depth32Float 1×1 D2Array fallback for shadow-less frames.
fn create_1x1_depth_d2array(device: &wgpu::Device, label: &str) -> Tracked<wgpu::TextureView> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    }))
}

/// Depth32Float 1×1×6 CubeArray fallback for point-shadow-less frames.
fn create_1x1_depth_cube_array(device: &wgpu::Device, label: &str) -> Tracked<wgpu::TextureView> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    Tracked::new(texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::CubeArray),
        ..Default::default()
    }))
}
