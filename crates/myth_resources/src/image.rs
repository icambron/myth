//! Raw image data storage.
//!
//! [`Image`] is a pure CPU-side data container holding pixel bytes and
//! format metadata. It owns no GPU resources and carries no graphics API
//! dependencies — all format/dimension types are engine-native enums that
//! map to `wgpu` equivalents only at GPU upload time.
//!
//! # Key types
//!
//! * [`PixelFormat`] — describes the in-memory pixel layout without any
//!   colour-space semantics (sRGB vs Linear is decided by [`Texture`]).
//! * [`ImageDimension`] — spatial dimensionality (1D / 2D / 3D).
//! * [`ColorSpace`] — rendering intent; lives on [`Texture`] and is combined
//!   with `PixelFormat` at GPU upload time to produce the final
//!   `wgpu::TextureFormat`.
//!
//! # Storage modes
//!
//! * [`Image::new`] creates a static image or an empty placeholder when data is
//!   not resident yet.
//! * [`Image::new_dynamic`] creates a fixed-capacity CPU buffer for streaming
//!   workloads such as video frames or camera feeds.
//! * Asset version tracking lives in the asset storage layer. When an image is
//!   stored in `AssetStorage`, use the storage's dynamic update API so render
//!   synchronization stays coherent.

use parking_lot::{RwLock, RwLockReadGuard};
use std::ops::Deref;
use uuid::Uuid;

// ────────────────────────────────────────────────────────────────────────────
// Engine-native pixel format
// ────────────────────────────────────────────────────────────────────────────

/// GPU-independent pixel format describing the in-memory byte layout.
///
/// This enum intentionally does **not** encode colour-space information
/// (sRGB vs Linear). Colour-space is a rendering decision owned by
/// [`Texture::color_space`](crate::texture::Texture::color_space) and
/// resolved at GPU upload time via [`PixelFormat::to_wgpu`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 4 × 8-bit unsigned normalised RGBA.
    Rgba8Unorm,
    /// 4 × 16-bit IEEE 754 half-precision float RGBA.
    Rgba16Float,
    /// Single-channel 8-bit unsigned normalised.
    R8Unorm,
    /// BC block-compressed color or data, with 4x4 texels per block.
    Bc1,
    Bc3,
    Bc5,
    Bc6h,
    Bc7,
}

impl PixelFormat {
    /// Resolves the final `wgpu::TextureFormat` by combining the physical
    /// pixel layout with the requested colour space.
    ///
    /// Formats that have no sRGB variant (e.g. `Rgba16Float`) ignore the
    /// colour-space argument and always return the linear variant.
    #[must_use]
    pub fn to_wgpu(self, color_space: ColorSpace) -> wgpu::TextureFormat {
        match (self, color_space) {
            (Self::Rgba8Unorm, ColorSpace::Srgb) => wgpu::TextureFormat::Rgba8UnormSrgb,
            (Self::Rgba8Unorm, ColorSpace::Linear) => wgpu::TextureFormat::Rgba8Unorm,
            (Self::Rgba16Float, _) => wgpu::TextureFormat::Rgba16Float,
            (Self::R8Unorm, _) => wgpu::TextureFormat::R8Unorm,
            (Self::Bc1, ColorSpace::Srgb) => wgpu::TextureFormat::Bc1RgbaUnormSrgb,
            (Self::Bc1, _) => wgpu::TextureFormat::Bc1RgbaUnorm,
            (Self::Bc3, ColorSpace::Srgb) => wgpu::TextureFormat::Bc3RgbaUnormSrgb,
            (Self::Bc3, _) => wgpu::TextureFormat::Bc3RgbaUnorm,
            (Self::Bc5, _) => wgpu::TextureFormat::Bc5RgUnorm,
            (Self::Bc6h, _) => wgpu::TextureFormat::Bc6hRgbUfloat,
            (Self::Bc7, ColorSpace::Srgb) => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
            (Self::Bc7, _) => wgpu::TextureFormat::Bc7RgbaUnorm,
        }
    }

    /// Bytes occupied by a single texel (or compressed block) in this format.
    #[inline]
    #[must_use]
    pub const fn block_copy_size(self) -> u32 {
        match self {
            Self::Rgba8Unorm => 4,
            Self::Rgba16Float | Self::Bc1 => 8,
            Self::R8Unorm => 1,
            Self::Bc3 | Self::Bc5 | Self::Bc6h | Self::Bc7 => 16,
        }
    }

    /// Dimensions of one storage block in texels.
    #[must_use]
    pub const fn block_dimensions(self) -> (u32, u32) {
        match self {
            Self::Bc1 | Self::Bc3 | Self::Bc5 | Self::Bc6h | Self::Bc7 => (4, 4),
            _ => (1, 1),
        }
    }

    /// Size of a two-dimensional mip level, including partial edge blocks.
    #[must_use]
    pub fn mip_byte_size(self, width: u32, height: u32) -> Option<usize> {
        let (bw, bh) = self.block_dimensions();
        (width.div_ceil(bw) as usize)
            .checked_mul(height.div_ceil(bh) as usize)?
            .checked_mul(self.block_copy_size() as usize)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Engine-native image dimension
// ────────────────────────────────────────────────────────────────────────────

/// Spatial dimensionality of an image's texel grid.
///
/// Maps 1:1 to `wgpu::TextureDimension` but lives in engine-native code
/// so that [`Image`] carries no `wgpu` dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageDimension {
    D1,
    D2,
    D3,
}

impl ImageDimension {
    /// Converts to the corresponding `wgpu::TextureDimension`.
    #[inline]
    #[must_use]
    pub fn to_wgpu(self) -> wgpu::TextureDimension {
        match self {
            Self::D1 => wgpu::TextureDimension::D1,
            Self::D2 => wgpu::TextureDimension::D2,
            Self::D3 => wgpu::TextureDimension::D3,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Colour space
// ────────────────────────────────────────────────────────────────────────────

/// Describes how pixel data should be interpreted in the GPU shader.
///
/// This is a **rendering intent** — the same physical pixel bytes can be
/// displayed as sRGB (e.g. base colour map) or treated as linear data
/// (e.g. roughness / normal map) depending on the material's needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// Non-linear sRGB encoding. Hardware performs automatic sRGB → linear
    /// conversion on texture fetch.
    Srgb,
    /// Linear encoding. No conversion is applied.
    Linear,
}

// ────────────────────────────────────────────────────────────────────────────
// Image
// ────────────────────────────────────────────────────────────────────────────

/// Backing storage for CPU-side image bytes.
///
/// Static images keep the existing lock-free path. Dynamic images pre-allocate
/// their CPU buffer and reuse it behind an internal lock so callers can stream
/// new bytes without allocating a replacement [`Image`].
#[derive(Debug)]
pub enum ImageStorage {
    /// Placeholder image with no resident CPU data yet.
    Empty,
    /// Immutable image payload.
    Static(Vec<u8>),
    /// Mutable image payload for high-frequency updates.
    Dynamic(RwLock<Vec<u8>>),
}

/// Borrowed image bytes.
///
/// Static images expose a plain slice. Dynamic images hold a read lock for the
/// duration of the borrow, allowing upload code to read without cloning.
pub enum ImageDataRef<'a> {
    Static(&'a [u8]),
    Dynamic(RwLockReadGuard<'a, Vec<u8>>),
}

impl Deref for ImageDataRef<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Static(data) => data,
            Self::Dynamic(guard) => guard.as_slice(),
        }
    }
}

impl AsRef<[u8]> for ImageDataRef<'_> {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

/// Failure modes for in-place dynamic image updates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicImageError {
    MissingData,
    NotDynamic,
    SizeMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for DynamicImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingData => f.write_str("image has no resident CPU data"),
            Self::NotDynamic => f.write_str("image was not created with Image::new_dynamic"),
            Self::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "image byte length mismatch: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for DynamicImageError {}

/// CPU-side image data.
///
/// Carries pixel bytes plus the minimal metadata the GPU uploader needs
/// (size, format, dimension). Dynamic images keep their CPU buffer behind a
/// small internal lock so callers can update bytes in place while the asset
/// storage continues to own version tracking.
///
/// Colour-space semantics are intentionally absent — the same image data can
/// be uploaded as sRGB or Linear depending on the [`Texture`] that references
/// it.
#[derive(Debug)]
pub struct Image {
    pub uuid: Uuid,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    mip_level_count: u32,
    pub dimension: ImageDimension,
    pub format: PixelFormat,
    /// Raw pixel bytes.
    storage: ImageStorage,
}

impl Image {
    /// Creates a new image with the given dimensions and optional pixel data.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        depth_or_array_layers: u32,
        dimension: ImageDimension,
        format: PixelFormat,
        data: Option<Vec<u8>>,
    ) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            width,
            height,
            depth: depth_or_array_layers,
            mip_level_count: 1,
            dimension,
            format,
            storage: match data {
                Some(data) => ImageStorage::Static(data),
                None => ImageStorage::Empty,
            },
        }
    }

    /// Creates a static 2D image with tightly packed mip levels, largest first.
    /// Rejects zero dimensions, impossible mip counts, and mismatched payloads.
    pub fn from_mip_levels(
        width: u32,
        height: u32,
        format: PixelFormat,
        levels: &[&[u8]],
    ) -> Result<Self, &'static str> {
        if width == 0
            || height == 0
            || levels.is_empty()
            || levels.len() > (width.max(height).ilog2() + 1) as usize
        {
            return Err("invalid mip dimensions or count");
        }
        let mut data = Vec::new();
        for (level, bytes) in levels.iter().enumerate() {
            let expected = format
                .mip_byte_size((width >> level).max(1), (height >> level).max(1))
                .ok_or("mip size overflow")?;
            if bytes.len() != expected {
                return Err("mip payload size does not match dimensions");
            }
            data.extend_from_slice(bytes);
        }
        let mut image = Self::new(width, height, 1, ImageDimension::D2, format, Some(data));
        image.mip_level_count = levels.len() as u32;
        Ok(image)
    }

    /// Number of levels carried in the pixel payload.
    #[must_use]
    pub const fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }

    /// Creates a dynamic image whose CPU buffer can be updated in place.
    ///
    /// The allocation backing `initial_data` is retained for the lifetime of
    /// the image. Subsequent dynamic updates must preserve the exact byte
    /// length so the storage can reuse the existing capacity without
    /// reallocating.
    #[must_use]
    pub fn new_dynamic(
        width: u32,
        height: u32,
        depth_or_array_layers: u32,
        dimension: ImageDimension,
        format: PixelFormat,
        initial_data: Vec<u8>,
    ) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            width,
            height,
            depth: depth_or_array_layers,
            mip_level_count: 1,
            dimension,
            format,
            storage: ImageStorage::Dynamic(RwLock::new(initial_data)),
        }
    }

    /// Returns the pixel format.
    #[inline]
    #[must_use]
    pub fn format(&self) -> PixelFormat {
        self.format
    }

    /// Returns the spatial dimensionality.
    #[inline]
    #[must_use]
    pub fn dimension(&self) -> ImageDimension {
        self.dimension
    }

    /// Returns `true` when the image has CPU-resident bytes.
    #[inline]
    #[must_use]
    pub fn has_data(&self) -> bool {
        !matches!(self.storage, ImageStorage::Empty)
    }

    /// Returns `true` when the image uses the dynamic update path.
    #[inline]
    #[must_use]
    pub fn is_dynamic(&self) -> bool {
        matches!(self.storage, ImageStorage::Dynamic(_))
    }

    /// Borrows the current image bytes without allocating.
    ///
    /// Static images return a plain slice. Dynamic images hold a read lock for
    /// the lifetime of the returned guard.
    #[must_use]
    pub fn data(&self) -> Option<ImageDataRef<'_>> {
        match &self.storage {
            ImageStorage::Empty => None,
            ImageStorage::Static(data) => Some(ImageDataRef::Static(data.as_slice())),
            ImageStorage::Dynamic(data) => Some(ImageDataRef::Dynamic(data.read())),
        }
    }

    /// Executes `f` with the current image bytes, if present.
    ///
    /// This is the preferred read path for upload code that only needs a
    /// temporary byte slice and wants to avoid naming the guard type.
    #[inline]
    pub fn with_data<R>(&self, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
        self.data().map(|data| f(data.as_ref()))
    }

    /// Overwrites a dynamic image buffer in place without reallocating.
    ///
    /// This mutates only the CPU-side bytes. It does not bump any asset
    /// version counters. When the image is managed by the asset system, call
    /// the storage-layer dynamic update API so render synchronization can see
    /// the change.
    pub fn update_dynamic_data(&self, new_data: &[u8]) -> Result<(), DynamicImageError> {
        match &self.storage {
            ImageStorage::Empty => Err(DynamicImageError::MissingData),
            ImageStorage::Static(_) => Err(DynamicImageError::NotDynamic),
            ImageStorage::Dynamic(data) => {
                let mut buffer = data.write();
                if buffer.len() != new_data.len() {
                    return Err(DynamicImageError::SizeMismatch {
                        expected: buffer.len(),
                        actual: new_data.len(),
                    });
                }
                buffer.copy_from_slice(new_data);
                Ok(())
            }
        }
    }

    /// Creates a 1×1 RGBA8 image with the specified colour.
    #[must_use]
    pub fn solid_color(rgba: [u8; 4]) -> Self {
        Self::new(
            1,
            1,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(rgba.to_vec()),
        )
    }

    /// Creates a checkerboard pattern image.
    #[must_use]
    pub fn checkerboard(width: u32, height: u32, cell_size: u32) -> Self {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let is_white = ((x / cell_size) + (y / cell_size)).is_multiple_of(2);
                let c = if is_white { 255u8 } else { 80u8 };
                data.extend_from_slice(&[c, c, c, 255]);
            }
        }
        Self::new(
            width,
            height,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(data),
        )
    }
}

#[cfg(test)]
mod mip_tests {
    use super::{Image, PixelFormat};

    #[test]
    fn partial_edge_blocks_and_tail_mips_retain_complete_blocks() {
        assert_eq!(PixelFormat::Bc1.mip_byte_size(7, 5), Some(32));
        assert_eq!(PixelFormat::Bc5.mip_byte_size(1, 1), Some(16));
        let image = Image::from_mip_levels(
            8,
            4,
            PixelFormat::Bc1,
            &[&[1; 16], &[2; 8], &[3; 8], &[4; 8]],
        )
        .unwrap();
        assert_eq!(image.mip_level_count(), 4);
        assert_eq!(image.data().unwrap().len(), 40);
    }

    #[test]
    fn malformed_chains_fail_before_upload() {
        assert!(Image::from_mip_levels(0, 4, PixelFormat::Bc1, &[&[0; 8]]).is_err());
        assert!(Image::from_mip_levels(4, 4, PixelFormat::Bc1, &[]).is_err());
        assert!(Image::from_mip_levels(4, 4, PixelFormat::Bc1, &[&[0; 7]]).is_err());
        assert!(Image::from_mip_levels(1, 1, PixelFormat::Bc1, &[&[0; 8], &[0; 8]]).is_err());
    }
}
