#[test]
fn radius_lights_survive_gpu_culling() {
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("light attenuation regression requires a GPU adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("GPU device");
        let source = format!(
            "{}\n{}",
            include_str!("../src/pipeline/shaders/core/light_attenuation.wgsl"),
            r#"
@group(0) @binding(0) var<storage, read_write> results: array<f32>;
@compute @workgroup_size(1)
fn main() {
    results[0] = getLightCullingDistance(2.2, 10.0, -2.0, 0.005);
    results[1] = getDistanceAttenuation(0.0, 10.0, -2.0);
    results[2] = getDistanceAttenuation(5.0, 10.0, -2.0);
    results[3] = getDistanceAttenuation(10.0, 10.0, -2.0);
    results[4] = getLightCullingDistance(0.0, 10.0, -2.0, 0.005);
    results[5] = getLightCullingDistance(0.001, 10.0, -2.0, 0.005);
    results[6] = getLightCullingDistance(0.5, 100.0, 2.0, 0.005);
    results[7] = getLightCullingDistance(2.2, 0.0, -2.0, 0.005);
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Light attenuation regression"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 32);
        queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = readback.slice(..).get_mapped_range();
        let actual: &[f32] = bytemuck::cast_slice(&mapped);
        for (index, expected) in [10.0_f32, 1.0, 0.5625, 0.0, -1.0, -1.0, 10.0, -1.0]
            .into_iter()
            .enumerate()
        {
            assert!(
                (actual[index] - expected).abs() < 0.0001,
                "GPU result {index}: {} != {expected}",
                actual[index]
            );
        }
    });
}
