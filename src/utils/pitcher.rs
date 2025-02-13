use pitch_detection::detector::mcleod::McLeodDetector;
use pitch_detection::detector::PitchDetector;

/// 提取基础周期波形工具函数
/// 
/// # 参数
/// - `signal`: 原始时域波形数据（&[f64]）
/// - `sample_rate`: 采样率（单位Hz）
/// - `window_size`: 分析窗口长度（推荐1024/2048等2的幂）
/// - `power_thresh`: 能量阈值（默认5.0）
/// - `clarity_thresh`: 清晰度阈值（默认0.6）
///
/// # 返回
/// - Some(Vec<f64>): 基础周期波形数据
/// - None: 检测失败或输入不合法
pub fn extract_fundamental_waveform(
    signal: &[f64],
    sample_rate: usize,
    window_size: usize,
    power_thresh: f64,
    clarity_thresh: f64,
) -> Option<Vec<f64>> {
    // 参数合法性检查
    if signal.is_empty() || sample_rate == 0 || window_size == 0 {
        panic!("Invalid input");
    }

    // 创建检测器实例
    let padding = window_size / 2;
    let mut detector = McLeodDetector::new(window_size, padding);

    // 截取分析窗口（取信号中间部分以避开边缘效应）
    let start_idx = signal.len().saturating_sub(window_size) / 2;
    let frame = signal.get(start_idx..start_idx + window_size)?;

    // 基频检测
    let pitch = detector.get_pitch(frame, sample_rate, power_thresh, clarity_thresh)?;

    // 计算单周期样本数
    let samples_per_period = (sample_rate as f64 / pitch.frequency).round() as usize;
    if samples_per_period == 0 || samples_per_period > signal.len() {
        return None;
    }

    // 寻找相位起点（过零点）
    let start_pos = frame
        .windows(2)
        .position(|w| w[0] <= 0.0 && w[1] > 0.0)
        .unwrap_or(0);

    // 截取完整周期
    signal
        .get(start_pos..start_pos + samples_per_period)
        .map(|s| s.to_vec())
}

/// 测试用例
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_wave() {
        // 生成440Hz正弦波（3个周期）
        let sample_rate = 48000;
        let freq = 440.0;
        let duration = 3.0 / freq;
        let total_samples = (duration * sample_rate as f64).round() as usize;
        
        let signal: Vec<_> = (0..total_samples)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate as f64).sin())
            .collect();

        // 调用提取函数
        let fundamental = extract_fundamental_waveform(&signal, sample_rate, 2048, 5.0, 0.7);

        assert!(fundamental.is_some());
        assert_eq!(fundamental.unwrap().len(), (sample_rate as f64 / freq).round() as usize);
    }
}