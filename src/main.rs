use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Float;
use rustfft::{FftDirection, FftPlanner};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use vizia::prelude::*;
use vizia::vg;

mod utils;

const SIZE: usize = 1024;
// const PADDING: usize = SIZE / 2;
const POWER_THRESHOLD: f64 = 5.0;
const CLARITY_THRESHOLD: f64 = 0.7;

fn main() -> Result<(), ApplicationError> {
    let audio_buffer = Arc::new(Mutex::new(Vec::new()));
    let buffer_clone = audio_buffer.clone();
    let host = cpal::default_host();
    let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);

    let device = host.default_output_device().unwrap();
    let mut supported_configs_range = device.supported_output_configs().unwrap();
    let supported_config = supported_configs_range
        .next()
        .unwrap()
        .with_max_sample_rate();
    let sample_format = supported_config.sample_format();
    let config: cpal::StreamConfig = supported_config.into();
    let sample_rate = config.sample_rate.0;

    // 音频输入线程？启动！
    thread::spawn(move || {
        let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buffer = buffer_clone.lock().unwrap();
            // Downsample 但是 factor 10
            let downsampled_data: Vec<f32> = data.iter().step_by(1).copied().collect();
            let buffer_len = buffer.len();
            if buffer_len > 1024 {
                buffer.drain(..buffer_len - 1024);
                // buffer.clear();
            }
            buffer.extend(downsampled_data);
        };

        let stream = match sample_format {
            _ => device.build_input_stream(&config, input_data_fn, err_fn, None),
        }
        .unwrap();

        stream.play().unwrap();

        loop {
            std::thread::sleep(std::time::Duration::from_millis(1000));
        }
    });

    Application::new(move |cx| {
        AppData {
            waveform: audio_buffer.clone(),
        }
        .build(cx);

        HStack::new(cx, |cx| {
            WaveformView::new(cx, AppData::waveform, sample_rate)
                .size(Pixels(800.0))
                .height(Pixels(150.0));

            SpectrumView::new(cx, AppData::waveform, sample_rate as f32, 20000.0)
                .size(Pixels(800.0))
                .height(Pixels(300.0));
        });

        let timer = cx.add_timer(Duration::from_millis(33), None, |cx, _| {
            cx.emit(AppEvent::UpdateWaveform);
        });

        cx.start_timer(timer);
    })
    .inner_size((1600, 300))
    .title("sout meter input test")
    .run()
}

#[derive(Lens)]
struct AppData {
    waveform: Arc<Mutex<Vec<f32>>>,
}

impl Model for AppData {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|app_event, _| match app_event {
            AppEvent::UpdateWaveform => {
                cx.needs_redraw();
            }
        });
    }
}

pub enum AppEvent {
    UpdateWaveform,
}

pub struct WaveformView<L: Lens<Target = Arc<Mutex<Vec<f32>>>>> {
    input_data: L,
    sample_rate: u32,
}

impl<L> WaveformView<L>
where
    L: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    pub fn new(cx: &mut Context, data: L, sample_rate: u32) -> Handle<Self> {
        Self {
            input_data: data,
            sample_rate: sample_rate,
        }
        .build(cx, |_| {})
        .bind(data, |mut handle, _| handle.needs_redraw())
    }
}

impl<L: Lens<Target = Arc<Mutex<Vec<f32>>>>> View for WaveformView<L> {
    fn draw(&self, cx: &mut DrawContext, canvas: &Canvas) {
        let mut data = self.input_data.get(cx).lock().unwrap().clone();
        let bounds = cx.bounds();

        let signal: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        let period_samples = bounds.w as usize; // 目标点数，随窗口宽度变化

        if !signal.is_empty() {
            let new_data = utils::pitcher::extract_fundamental_waveform(
                &signal,
                self.sample_rate.try_into().unwrap(),
                SIZE,
                POWER_THRESHOLD,
                CLARITY_THRESHOLD,
            );

            if let Some(omg) = new_data {
                data = omg.iter().map(|&x| x as f32).collect();
            }
        }

        // 将波形数据插值到目标宽度对应的点数
        data = if data.is_empty() {
            vec![0.0; period_samples]
        } else {
            interpolate_data(&data, period_samples)
        };

        // 创建绘图路径
        let mut path = vg::Path::new();
        let step = if period_samples > 1 {
            bounds.w / (period_samples - 1) as f32
        } else {
            0.0
        };

        for (i, &value) in data.iter().enumerate() {
            let x = bounds.x + i as f32 * step;
            let y = bounds.y + bounds.h * (1.0 - value);
            let point = vg::Point::new(x, y);

            if i == 0 {
                path.move_to(point);
            } else {
                path.line_to(point);
            }
        }

        // 设置画笔并绘制
        let mut paint = vg::Paint::default();
        paint.set_style(vg::PaintStyle::Stroke);
        canvas.draw_path(&path, &paint);
    }
}

struct SpectrumView<L>
where
    L: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    input_data: L,
    fft: Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,
    fft_buffer: Vec<Complex<f32>>,
    sample_rate: f32,
    max_freq: f32,
    smoothing_buffer: Arc<Mutex<VecDeque<Vec<f32>>>>,
    buffer_size: usize,
}

impl<L> SpectrumView<L>
where
    L: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    pub fn new(cx: &mut Context, data: L, sample_rate: f32, max_freq: f32) -> Handle<Self> {
        let fft_size = 1024;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        // 生成汉宁窗 -----------------------------------------------------
        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32).cos())
            })
            .collect();

        Self {
            input_data: data,
            fft,
            window,
            fft_buffer: vec![Complex::default(); fft_size],
            sample_rate,
            max_freq: max_freq.min(sample_rate / 2.0 * 0.99), // 保留1%余量
            smoothing_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(8))),
            buffer_size: 8,
        }
        .build(cx, |_| {})
        .bind(data, |mut handle, _| handle.needs_redraw())
    }
}

impl<L: Lens<Target = Arc<Mutex<Vec<f32>>>>> View for SpectrumView<L> {
    fn draw(&self, cx: &mut DrawContext, canvas: &Canvas) {
        let waveform = self.input_data.get(cx).lock().unwrap().clone();
        let bounds = cx.bounds();
        let sample_rate = self.sample_rate;
        let max_freq = self.max_freq;

        // 准备FFT输入数据 -------------------------------------------------
        let mut fft_input = self.fft_buffer.clone();
        let fft_size = 1024;

        // 应用汉宁窗 -----------------------------------------------------
        waveform
            .iter()
            .take(fft_size)
            .chain(std::iter::repeat(&0.0)) // 不足补零
            .zip(&self.window) // 这里使用预先生成的汉宁窗
            .take(fft_size)
            .enumerate()
            .for_each(|(i, (s, w))| {
                fft_input[i] = Complex {
                    re: s * w, // 应用汉宁窗
                    im: 0.0,
                };
            });

        // 执行FFT变换 -----------------------------------------------------
        self.fft.process(&mut fft_input);

        let mut magnitudes: Vec<f32> = fft_input
            .iter()
            .take(512)
            .enumerate()
            .map(|(i, c)| {
                let scale = if i == 0 || i == 512 - 1 {
                    2.0 / 1.63
                } else {
                    2.0
                };
                (c.norm() * scale).max(1e-6)
            })
            .collect();

        let mut smoothing_buffer = self.smoothing_buffer.lock().unwrap();

        // 平滑处理
        smoothing_buffer.push_back(magnitudes);
        if smoothing_buffer.len() > self.buffer_size {
            smoothing_buffer.pop_front();
        }

        let spectrum_db: Vec<f32> = (0..512)
            .map(|i| {
                let avg =
                    smoothing_buffer.iter().map(|v| v[i]).sum::<f32>() / self.buffer_size as f32;
                20.0 * avg.log10().clamp(-4.5, 1.0) // -90dB到+20dB
            })
            .collect();

        // 计算显示参数
        let nyquist = sample_rate / 2.0;
        let max_bin = ((max_freq / nyquist) * (fft_size / 2) as f32).floor() as usize;
        let bins_to_draw = max_bin.min(fft_size / 2);

        // 可视化配置
        let mut paint = vg::Paint::default();
        paint.set_style(vg::PaintStyle::Stroke);
        paint.set_color(vg::Color::from_rgb(100, 200, 100));
        paint.set_stroke_width((bounds.w / bins_to_draw as f32).max(1.0));

        let mut path = vg::Path::new();

        // 三段式频率轴参数
        let low_cutoff = 200.0; // 低频/中频分界
        let mid_cutoff = 2000.0; // 中频/高频分界
        let log_low_base = (low_cutoff + 1.0).ln(); // 低频段使用自然对数
        let log_mid_base = (mid_cutoff / low_cutoff).log10(); // 中频段使用log10
        let log_high_base = (max_freq / mid_cutoff).log10(); // 高频段使用log10

        spectrum_db.iter().enumerate().for_each(|(i, db)| {
            // 计算实际频率
            let freq = i as f32 * nyquist / bins_to_draw as f32;

            // 三段式对数映射
            let x_ratio = if freq <= low_cutoff {
                // 低频段（0-200Hz）：自然对数映射
                (freq + 1.0).ln() / log_low_base * 0.3
            } else if freq <= mid_cutoff {
                // 中频段（200-2000Hz）：log10映射
                0.3 + (freq.log10() - low_cutoff.log10()) / log_mid_base * 0.4
            } else {
                // 高频段（2000-20000Hz）：log10映射
                0.7 + (freq.log10() - mid_cutoff.log10()) / log_high_base * 0.3
            };

            let x = bounds.x + x_ratio.clamp(0.0, 1.0) * bounds.w;

            // 垂直映射（-90dB到+20dB）
            let min_db = -100.0;
            let max_db = 100.0;
            let db_normalized = (db - min_db) / (max_db - min_db);
            let y = bounds.y + bounds.h * (1.0 - db_normalized.clamp(0.0, 1.0));

            path.move_to(vg::Point::new(x, y));
            path.line_to(vg::Point::new(x, bounds.y + bounds.h));
        });

        canvas.draw_path(&path, &paint);
    }
}

// 线性插值函数：将数据扩展到目标长度
fn interpolate_data(data: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 || data.is_empty() {
        return vec![0.0; target_len];
    }
    let original_len = data.len();
    if original_len == target_len {
        return data.to_vec();
    }
    let mut result = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let pos = i as f32 * (original_len - 1) as f32 / (target_len - 1) as f32;
        let index = pos as usize;
        if index >= original_len - 1 {
            result.push(data[original_len - 1]);
        } else {
            let fraction = pos - index as f32;
            let a = data[index];
            let b = data[index + 1];
            result.push(a + fraction * (b - a));
        }
    }
    result
}
