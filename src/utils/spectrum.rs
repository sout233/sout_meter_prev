use rustfft::{FftPlanner, num_complex::Complex, FftDirection};
use std::f32::consts::LN_2;

// 时域转频域函数
pub fn time_to_frequency(audio_waveform: &[f32], floor_db: f32) -> Vec<f32> {
    let n = audio_waveform.len();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(n, FftDirection::Forward);
    
    let mut buffer: Vec<Complex<f32>> = audio_waveform.iter()
        .map(|&x| Complex { re: x, im: 0.0 })
        .collect();
    
    fft.process(&mut buffer);
    
    let num_freq_bins = n / 2 + 1;
    let mut magnitudes = Vec::with_capacity(num_freq_bins);
    
    for i in 0..num_freq_bins {
        let amp = buffer[i].norm();
        let db = 20.0 * (amp + 1e-12).log10();
        magnitudes.push(db.max(floor_db));
    }
    
    magnitudes
}

// 频域斜率优化
pub fn optimize_slope(
    freq_db: &[f32],
    sampling_rate: f32,
    target_slope_db_per_oct: f32,
    reference_freq: f32
) -> Vec<f32> {
    let n_original = 2 * (freq_db.len() - 1);
    let mut adjusted_db = Vec::with_capacity(freq_db.len());
    
    for (i, &db) in freq_db.iter().enumerate() {
        let freq = i as f32 * sampling_rate / n_original as f32;
        
        if freq == 0.0 {
            adjusted_db.push(db);
            continue;
        }
        
        let octaves = (freq / reference_freq).log2();
        let adjustment = target_slope_db_per_oct * octaves;
        adjusted_db.push(db + adjustment);
    }
    
    adjusted_db
}

// 频域x轴优化（三段式对数轴）
pub fn optimize_xaxis(
    freq_db: &[f32],
    sampling_rate: f32,
    low_cutoff: f32,
    mid_cutoff: f32,
    low_points: usize,
    mid_points: usize,
    high_points: usize,
) -> (Vec<f32>, Vec<f32>) {
    let nyquist = sampling_rate / 2.0;
    let n_original = 2 * (freq_db.len() - 1);
    
    // 生成原始频率轴
    let original_freqs: Vec<f32> = (0..freq_db.len())
        .map(|i| i as f32 * sampling_rate / n_original as f32)
        .collect();
    
    // 生成新的三段式对数频率轴
    let mut new_freqs = Vec::new();
    
    // 低频段（跳过0Hz）
    if low_points > 0 {
        let low_start = original_freqs[1].max(1e-3); // 从第一个非零频率开始
        append_logspace(low_start, low_cutoff, low_points, &mut new_freqs);
    }
    
    // 中频段
    if mid_points > 0 {
        append_logspace(low_cutoff, mid_cutoff, mid_points, &mut new_freqs);
    }
    
    // 高频段
    if high_points > 0 {
        append_logspace(mid_cutoff, nyquist, high_points, &mut new_freqs);
    }
    
    // 插值处理
    let interpolated_db = linear_interpolate(&original_freqs, freq_db, &new_freqs);
    
    (new_freqs, interpolated_db)
}

// 生成对数间隔频率点
fn append_logspace(start: f32, end: f32, points: usize, result: &mut Vec<f32>) {
    if start >= end || points == 0 {
        return;
    }
    
    let log_start = start.ln();
    let log_end = end.ln();
    let step = (log_end - log_start) / (points - 1) as f32;
    
    for i in 0..points {
        let freq = (log_start + step * i as f32).exp();
        result.push(freq);
    }
}

// 线性插值
fn linear_interpolate(x_old: &[f32], y_old: &[f32], x_new: &[f32]) -> Vec<f32> {
    let mut y_new = Vec::with_capacity(x_new.len());
    
    for &x in x_new {
        match x_old.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
            Ok(pos) => y_new.push(y_old[pos]),
            Err(pos) => {
                if pos == 0 {
                    y_new.push(y_old[0]);
                } else if pos >= x_old.len() {
                    y_new.push(*y_old.last().unwrap());
                } else {
                    let x0 = x_old[pos-1];
                    let x1 = x_old[pos];
                    let y0 = y_old[pos-1];
                    let y1 = y_old[pos];
                    let t = (x - x0) / (x1 - x0);
                    y_new.push(y0 + t * (y1 - y0));
                }
            }
        }
    }
    
    y_new
}