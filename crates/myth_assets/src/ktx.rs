//! Native-format 2D KTX2 images, retaining their authored mip payloads.

use ktx2::Format as F;
use myth_core::{AssetError, Error, Result};
use myth_resources::image::{Image, PixelFormat};

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Asset(AssetError::Format(format!("KTX2: {message}")))
}

pub(crate) fn decode(bytes: &[u8]) -> Result<Image> {
    let reader = ktx2::Reader::new(bytes).map_err(invalid)?;
    let header = reader.header();
    if header.pixel_height == 0
        || header.pixel_depth != 0
        || header.layer_count != 0
        || header.face_count != 1
        || header.supercompression_scheme.is_some()
    {
        return Err(invalid(
            "expected a native-format, unsupercompressed 2D image",
        ));
    }
    let format = match header.format {
        Some(F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB) => PixelFormat::Rgba8Unorm,
        Some(F::R8_UNORM) => PixelFormat::R8Unorm,
        Some(F::BC1_RGBA_UNORM_BLOCK | F::BC1_RGBA_SRGB_BLOCK) => PixelFormat::Bc1,
        Some(F::BC3_UNORM_BLOCK | F::BC3_SRGB_BLOCK) => PixelFormat::Bc3,
        Some(F::BC5_UNORM_BLOCK) => PixelFormat::Bc5,
        Some(F::BC6H_UFLOAT_BLOCK) => PixelFormat::Bc6h,
        Some(F::BC7_UNORM_BLOCK | F::BC7_SRGB_BLOCK) => PixelFormat::Bc7,
        other => return Err(invalid(format!("unsupported format {other:?}"))),
    };
    let (bw, bh) = format.block_dimensions();
    if header.pixel_width % bw != 0 || header.pixel_height % bh != 0 {
        return Err(invalid(
            "base dimensions must be multiples of the storage block",
        ));
    }
    let levels: Vec<_> = reader.levels().map(|level| level.data).collect();
    Image::from_mip_levels(header.pixel_width, header.pixel_height, format, &levels)
        .map_err(invalid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        let header = ktx2::Header {
            format: Some(ktx2::Format::BC1_RGBA_UNORM_BLOCK),
            type_size: 1,
            pixel_width: 4,
            pixel_height: 4,
            pixel_depth: 0,
            layer_count: 0,
            face_count: 1,
            level_count: 2,
            supercompression_scheme: None,
            index: ktx2::Index {
                dfd_byte_offset: 128,
                dfd_byte_length: 4,
                kvd_byte_offset: 0,
                kvd_byte_length: 0,
                sgd_byte_offset: 0,
                sgd_byte_length: 0,
            },
        };
        let mut bytes = header.as_bytes().to_vec();
        for offset in [144_u64, 136] {
            bytes.extend_from_slice(&offset.to_le_bytes());
            bytes.extend_from_slice(&8_u64.to_le_bytes());
            bytes.extend_from_slice(&8_u64.to_le_bytes());
        }
        bytes.resize(136, 0);
        bytes.extend_from_slice(&[2; 8]);
        bytes.extend_from_slice(&[1; 8]);
        bytes
    }

    #[test]
    fn authored_levels_are_uploaded_largest_first_without_decoding() {
        let image = decode(&fixture()).unwrap();
        assert_eq!(image.format, PixelFormat::Bc1);
        assert_eq!(image.mip_level_count(), 2);
        assert_eq!(
            &*image.data().unwrap(),
            &[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        );
    }

    #[test]
    fn truncated_and_wrong_sized_levels_are_rejected() {
        let mut bytes = fixture();
        bytes.pop();
        assert!(decode(&bytes).is_err());
        let mut bytes = fixture();
        bytes[88..96].copy_from_slice(&7_u64.to_le_bytes());
        assert!(decode(&bytes).is_err());
    }
}
