//! Persistent asset mipmap generation.
//!
//! Texture upload is owned by `ResourceManager`, which queues a [`MipmapRequest`]
//! whenever a mipmapped image's level 0 has been written. The generation itself
//! happens here, on its own command encoder submitted during prepare.
//!
//! This used to be folded into the frame graph so it could share the frame's
//! command encoder. That is safe for the *transient* mip generation the other
//! passes do — those read a texture some earlier pass in the same graph wrote,
//! so the graph has an edge to order them by. An asset's level 0 arrives by
//! `Queue::write_texture` instead, and the texture it lands in is external to
//! the graph, so the pass had no edges at all: `mark_side_effect` kept it from
//! being culled, but nothing constrained where in the execution queue it ran.
//! Most textures survived that, because nothing samples them until a later
//! frame. One per session did not, and because `mark_complete` latches, the
//! bad chain was never regenerated — it stayed wrong for the life of the page.
//!
//! The symptom that found it: a pool's ripple flipbook cycles 48 mipmapped
//! textures at 6 Hz, so one of the 48 had a garbage chain and the water went
//! flat for a sixth of a second once every loop — level 0 was intact, so it
//! only showed at distance, where the upper levels are what gets sampled.
//!
//! Generating here costs one extra submit per batch, and only while textures
//! are still arriving: [`GpuImage::mipmap_request`] returns `None` once a
//! chain has been generated.
//!
//! [`GpuImage::mipmap_request`]: crate::core::gpu::GpuImage::mipmap_request

use crate::core::gpu::MipmapGenerator;
use crate::graph::core::ExtractContext;

pub struct MipmapFeature {
    generator: MipmapGenerator,
}

impl MipmapFeature {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            generator: MipmapGenerator::new(device),
        }
    }

    /// The generator, for the graph passes that mip transient textures.
    #[inline]
    #[must_use]
    pub fn generator(&self) -> &MipmapGenerator {
        &self.generator
    }

    /// Generates every mip chain queued since the last call.
    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        let requests = ctx.resource_manager.take_mipmap_requests();
        if requests.is_empty() {
            return;
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Asset Mipmap Generation"),
            });
        for request in &requests {
            self.generator.ensure_pipeline(ctx.device, request.format);
            self.generator
                .generate(ctx.device, &mut encoder, &request.texture);
        }
        ctx.queue.submit(Some(encoder.finish()));

        // Only once the work is submitted, so a chain is never marked done on
        // the strength of commands that were merely recorded.
        for request in &requests {
            request.mark_complete();
        }
    }
}
