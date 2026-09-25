use glam::{Vec2, Vec3, Vec4};
use myth_macros::myth_material;

use crate::uniforms::Mat3Uniform;

#[myth_material(shader = "entry/main/phong", crate_path = "crate")]
pub struct PhongMaterial {
    /// Diffuse color.
    #[uniform(default = "Vec4::ONE")]
    pub color: Vec4,

    /// Specular color.
    #[uniform(default = "Vec3::splat(0.06667)")]
    pub specular: Vec3,

    /// Opacity value.
    #[uniform(default = "1.0")]
    pub opacity: f32,

    /// Emissive color.
    #[uniform(skip_builder)]
    pub emissive: Vec3,

    /// Emissive intensity.
    #[uniform(default = "1.0")]
    pub emissive_intensity: f32,

    /// Normal map scale.
    #[uniform(default = "Vec2::ONE")]
    pub normal_scale: Vec2,

    /// Reconstruct positive Z from an XY normal texture when set to one.
    #[uniform]
    pub normal_reconstruct_z: f32,

    /// Shininess factor.
    #[uniform(default = "30.0")]
    pub shininess: f32,

    /// Alpha test threshold.
    #[uniform]
    pub alpha_test: f32,

    /// A per-object colour laid over the diffuse map, recolouring it while
    /// keeping its relief and -- unlike `color`, which can only multiply --
    /// able to lighten. `rgb` is the overlay in linear space; `a` is the
    /// material's mean albedo luminance, which both scales the overlay to the
    /// map and, when zero, switches the whole thing off. Zero by default, so a
    /// material without one draws exactly as before.
    #[uniform(default = "Vec4::ZERO")]
    pub overlay: Vec4,

    /// The color map.
    #[texture]
    pub map: TextureSlot,

    /// The normal map.
    #[texture]
    pub normal_map: TextureSlot,

    /// The specular map.
    #[texture]
    pub specular_map: TextureSlot,

    /// The emissive map.
    #[texture]
    pub emissive_map: TextureSlot,
}

impl PhongMaterial {
    /// Creates a new Phong material with the given diffuse color.
    #[must_use]
    pub fn new(color: Vec4) -> Self {
        Self::from_uniforms(PhongUniforms {
            color,
            ..Default::default()
        })
    }

    /// Sets the emissive color and intensity (builder).
    #[must_use]
    pub fn with_emissive(self, color: Vec3, intensity: f32) -> Self {
        {
            let mut u = self.uniforms.write();
            u.emissive = color;
            u.emissive_intensity = intensity;
        }
        self
    }
}
