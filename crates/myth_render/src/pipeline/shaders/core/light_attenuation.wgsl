// Shared distance response for surface shading and clustered-light admission.
// Negative decay encodes a radius response, with exponent -decay.
fn getDistanceAttenuation(light_distance: f32, cutoff_distance: f32, decay_exponent: f32) -> f32 {
    if (decay_exponent < 0.0) {
        let relative_distance = light_distance / max(cutoff_distance, 0.0001);
        return pow(saturate(1.0 - relative_distance * relative_distance), -decay_exponent);
    }
    var distance_falloff = 1.0 / max(pow(light_distance, decay_exponent), 0.01);
    if (cutoff_distance > 0.0) {
        let relative_distance = light_distance / cutoff_distance;
        let squared = relative_distance * relative_distance;
        let window = saturate(1.0 - squared * squared);
        distance_falloff *= window * window;
    }
    return distance_falloff;
}

// Culling may overestimate influence but must retain visible contributions.
fn getLightCullingDistance(intensity: f32, range: f32, decay: f32, threshold: f32) -> f32 {
    if (range <= 0.0 || intensity <= 0.0) {
        return -1.0;
    }
    let closest_contribution = intensity * getDistanceAttenuation(0.0, range, decay);
    if (closest_contribution <= threshold) {
        return -1.0;
    }
    if (decay < 0.0) {
        // The radius response is finite at its origin and zero at its range.
        // Its complete support is a conservative bound, independent of exponent.
        return range;
    }
    // The inverse-square branch ignores the smoothing window conservatively.
    let safe_distance = pow(intensity / threshold, 1.0 / max(decay, 0.01));
    return min(safe_distance, range);
}
