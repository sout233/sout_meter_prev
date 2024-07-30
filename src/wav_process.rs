use hound;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

pub fn read_wav(file_path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(file_path).expect("Failed to open WAV file");
    let samples: Vec<f32> = reader.samples::<i32>()
        .map(|s| s.expect("Failed to read sample") as f32)
        .collect();

    // 归一化处理
    let max_sample = samples.iter().cloned().fold(f32::MIN, f32::max).abs();
    let normalized_samples = if max_sample > 0.0 {
        samples.iter().map(|&s| s / max_sample).collect()
    } else {
        samples
    };

    // Downsampling by taking every Nth sample
    let downsample_factor = 10; // Adjust this value to control the downsample rate
    normalized_samples.into_iter().step_by(downsample_factor).collect()
}


pub fn calculate_spectrum(samples: &[f32]) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(samples.len());
    let mut buffer: Vec<Complex<f32>> = samples.iter().map(|&s| Complex { re: s, im: 0.0 }).collect();
    fft.process(&mut buffer);
    
    buffer.iter().map(|c| c.norm()).collect()
}
