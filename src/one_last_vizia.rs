use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::num_traits::ConstZero;
use rustfft::{num_complex::Complex, FftPlanner};
use std::cell::RefCell;
use std::iter::Sum;
use std::sync::{Arc, Mutex, Once};
use std::thread;
use vizia::prelude::*;
use vizia::vg::{self, Color, FontId, Paint};

#[allow(unused)]
const FREQ_RANGE_START_HZ: f32 = 30.0;
#[allow(unused)]
const FREQ_RANGE_END_HZ: f32 = 22_000.0;
const LN_FREQ_RANGE_START_HZ: f32 = 3.4011974; // 30.0f32.ln();
const LN_FREQ_RANGE_END_HZ: f32 = 9.998797; // 22_000.0f32.ln();
const LN_FREQ_RANGE: f32 = LN_FREQ_RANGE_END_HZ - LN_FREQ_RANGE_START_HZ;
pub const MINUS_INFINITY_GAIN: f32 = 1e-5; // 10f32.powf(MINUS_INFINITY_DB / 20)

// 使用Once来确保只初始化一次
static FONT_LOADED: Once = Once::new();

fn main() -> Result<(), ApplicationError> {
    let audio_buffer = Arc::new(Mutex::new(Vec::new()));
    let spectrum_buffer = Arc::new(Mutex::new(Vec::new()));

    let buffer_clone = audio_buffer.clone();
    let spectrum_clone = spectrum_buffer.clone();

    // 音频输入线程？启动！
    thread::spawn(move || {
        let host = cpal::default_host();

        let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);

        let device = host.default_output_device().unwrap();
        let _ = host.devices().unwrap().map(|d| {
            println!("dev name: {}", d.name().unwrap());
        });

        let mut supported_configs_range = device.supported_output_configs().unwrap();
        let supported_config = supported_configs_range
            .next()
            .unwrap()
            .with_max_sample_rate();

        let sample_format = supported_config.sample_format();
        let config = supported_config.into();

        let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buffer = buffer_clone.lock().unwrap();

            // Downsample 但是 factor 10
            let downsampled_data: Vec<f32> = data.iter().step_by(10).copied().collect();
            buffer.extend(downsampled_data);

            let max_length: usize = 48000 / 10 * 10; // 48kHz 采样率下的 1 秒数据，再乘以10是为了存储最近的10秒数据
            if buffer.len() > max_length {
                let excess = buffer.len() - max_length;
                buffer.drain(0..excess);
            }

            // 更新spectrum cache
            // let mut spectrum = spectrum_clone.lock().unwrap();
            // let window_size = 1024;
            // if buffer.len() >= window_size {
            //     let mut fft_buffer: Vec<Complex<f32>> = buffer[buffer.len() - window_size..]
            //         .iter()
            //         .map(|&x| Complex { re: x, im: 0.0 })
            //         .collect();

            //     let num_bins = fft_buffer.len();

            //     let mut planner = FftPlanner::<f32>::new();
            //     let fft = planner.plan_fft_forward(window_size);
            //     fft.process(&mut fft_buffer);

            //     spectrum.clear();
            //     spectrum.extend(fft_buffer.iter().map(|c| c.norm()));
            // }
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
            spectrum: spectrum_buffer.clone(),
        }
        .build(cx);

        HStack::new(cx, |cx| {
            WaveformView::new(cx, AppData::waveform.clone())
                .size(Pixels(400.0))
                .height(Pixels(100.0));

            SpectrumView::new(cx, AppData::waveform.clone(), AppData::spectrum.clone())
                .size(Pixels(600.0))
                .height(Pixels(100.0))
                .top(Pixels(100.0));

            PeakView::new(cx, AppData::waveform.clone())
                .size(Pixels(100.0))
                .height(Pixels(100.0));
        })
        .on_press_down(|cx| {
            // cx.emit(WindowEvent::DragWindow);
        });

        let timer = cx.add_timer(Duration::from_millis(33), None, |cx, _| {
            cx.emit(AppEvent::UpdateWaveform);
        });

        cx.start_timer(timer);
    })
    .inner_size((600 + 400 + 100, 200))
    .always_on_top(true)
    .title("sout meter test")
    .decorations(false)
    .run()
}

#[derive(Lens)]
struct AppData {
    waveform: Arc<Mutex<Vec<f32>>>,
    spectrum: Arc<Mutex<Vec<f32>>>, // 疑似有点多余了
}

impl Model for AppData {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|app_event, _| match app_event {
            AppEvent::UpdateWaveform => {
                cx.needs_redraw(); // 重绘核心，小心勿动
            }
        });
    }
}

pub enum AppEvent {
    UpdateWaveform,
}

pub struct WaveformView<L: Lens<Target = Arc<Mutex<Vec<f32>>>>> {
    data: L,
}

impl<L> WaveformView<L>
where
    L: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    pub fn new(cx: &mut Context, data: L) -> Handle<Self> {
        Self { data }
            .build(cx, |_| {})
            .bind(data, |mut handle, _| handle.needs_redraw())
    }
}

impl<L> View for WaveformView<L>
where
    L: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let mut data = self.data.get(cx).lock().unwrap().clone();
        let bounds = cx.bounds();
        let max_samples = (bounds.w as usize).min(data.len());

        if data.len() > max_samples {
            data.drain(..data.len() - max_samples);
        }

        let mut path = vg::Path::new();
        let step = if data.len() > 1 {
            bounds.w / (data.len() - 1) as f32
        } else {
            0.0
        };

        for (i, &value) in data.iter().enumerate() {
            let x = bounds.x + i as f32 * step;
            let y = bounds.y + bounds.h * (1.0 - value);
            if i == 0 {
                path.move_to(x, y);
            } else {
                path.line_to(x, y);
            }
        }

        let paint = vg::Paint::color(Color::black().into()).with_line_width(2.0);
        canvas.stroke_path(&path, &paint);
    }
}

fn db_level(sample: f32) -> f32 {
    if sample == 0.0 {
        -100.0 // 静音时的分贝值
    } else {
        20.0 * sample.log10().abs()
    }
}

fn average_db_level(data: &[f32]) -> f32 {
    let sum_db: f32 = data.iter().map(|&sample| db_level(sample)).sum();
    sum_db / data.len() as f32
}

fn sample_to_db(sample: f32) -> f32 {
    // 转换样本值为dB值，假设sample值在[-1.0, 1.0]之间
    20.0 * sample.abs().log10()
}

fn calculate_average<T>(numbers: &[T]) -> f64
where
    T: Copy + Sum<T> + Into<f64>,
{
    if numbers.is_empty() {
        return 0.0;
    }

    let sum: T = numbers.iter().cloned().sum();
    let count = numbers.len() as f64;
    sum.into() / count
}

// peak视图
pub struct PeakView<L: Lens<Target = Arc<Mutex<Vec<f32>>>>> {
    data: L,
}

impl<L> PeakView<L>
where
    L: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    pub fn new(cx: &mut Context, data: L) -> Handle<Self> {
        Self { data }
            .build(cx, |_| {})
            .bind(data, |mut handle, _| handle.needs_redraw())
    }
}

impl<L> View for PeakView<L>
where
    L: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let data = self.data.get(cx).lock().unwrap().clone();
        let bounds = cx.bounds();

        // 计算当前电平值，使用最新的一个样本值
        // let current_sample = calculate_average(&data) as f32;
        let mut average = 0.0;
        for (i, d) in data.iter().enumerate() {
            average += d;
        }
        average /= data.len() as f32;
        average *= 10000.0;
        let current_sample = *data.last().unwrap_or(&0.0);
        if average > 0.0 {
            // println!("current_sample: {}", current_sample);
            // println!("average: {}", average);
        }
        // 将样本值转换为dB值
        let db_value = sample_to_db(current_sample) + 0.0;
        // 映射dB值到Y轴，假设dB值在[-60, 0]之间
        let y = bounds.y + bounds.h * (1.0 - (db_value + 60.0) / 60.0) + 10.0;

        // 绘制垂直线条
        let mut path = vg::Path::new();
        path.move_to(bounds.x + bounds.w / 2.0, bounds.y + bounds.h * 2.0);
        path.line_to(bounds.x + bounds.w / 2.0, y);

        let paint = vg::Paint::color(Color::black().into()).with_line_width(10.0);
        canvas.stroke_path(&path, &paint);

        let mut paint = vg::Paint::color(Color::rgb(20, 50, 100).into());

        load_font_if_needed(canvas, &mut paint);

        paint.set_text_align(vg::Align::Center);
        paint.set_font_size(15.0);

        let _ = canvas.fill_text(
            bounds.x + bounds.w / 2.0 + 30.0,
            y,
            db_value.round().to_string() + " dB",
            &paint,
        );
    }
}

// 频谱视图
pub struct SpectrumView<L1, L2>
where
    L1: Lens<Target = Arc<Mutex<Vec<f32>>>>,
    L2: Lens<Target = Arc<Mutex<Vec<f32>>>>, // where重灾区
{
    data: L1,
    spectrum: L2,
}

impl<L1, L2> SpectrumView<L1, L2>
where
    L1: Lens<Target = Arc<Mutex<Vec<f32>>>>,
    L2: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    pub fn new(cx: &mut Context, data: L1, spectrum: L2) -> Handle<Self> {
        Self { data, spectrum }
            .build(cx, |_| {})
            .bind(data, |mut handle, _| handle.needs_redraw())
            .bind(spectrum, |mut handle, _| handle.needs_redraw())
    }
}

impl<L1, L2> View for SpectrumView<L1, L2>
where
    L1: Lens<Target = Arc<Mutex<Vec<f32>>>>,
    L2: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let data = self.data.get(cx).lock().unwrap().clone();
        let spectrum = self.spectrum.get(cx).lock().unwrap().clone();
        let bounds = cx.bounds();

        if data == spectrum {
            return;
        }

        let window_size = 1024;
        if data.len() < window_size {
            return;
        }

        let window = hann_window(window_size);
        let mut buffer: Vec<Complex<f32>> = data[data.len() - window_size..]
            .iter()
            .zip(window.iter())
            .map(|(&sample, &window_coeff)| Complex {
                re: sample * window_coeff,
                im: 0.0,
            })
            .collect();

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        fft.process(&mut buffer);

        let num_bins = buffer.len();
        let magnitude: Vec<f32> = buffer[..num_bins].iter().map(|c| c.norm()).collect();

        // 计算 RMS 和 dB 值
        let rms = calculate_rms(&buffer);
        let rms_db = rms_to_db(rms);

        // 使用 RMS dB 值来动态调整高度
        let max_db = 0.0; // 设置为最大 dB 值
        let min_db = -77.0; // 设置为最小 dB 值
        let db_range = max_db - min_db;
        let normalized_db = |db: f32| (db - min_db) / db_range;

        let line_width = cx.scale_factor() * 1.5;
        let text_color: vg::Color = vg::Color::black();
        let bars_paint = vg::Paint::color(text_color).with_line_width(line_width);
        let mut lighter_text_color = text_color;
        lighter_text_color.r = (lighter_text_color.r + 0.25) / 1.25;
        lighter_text_color.g = (lighter_text_color.g + 0.25) / 1.25;
        lighter_text_color.b = (lighter_text_color.b + 0.25) / 1.25;

        let nyquist_hz = 48000.0 / 2.0;
        let bin_frequency = |bin_idx: f32| (bin_idx / (window_size as f32 / 2.0)) * nyquist_hz;
        let bin_t =
            |bin_idx: f32| (bin_frequency(bin_idx).ln() - LN_FREQ_RANGE_START_HZ) / LN_FREQ_RANGE;

        let max_amplitude = *magnitude
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);

        // 绘制频谱条
        let mut bars_path = vg::Path::new();
        for (bin_idx, &magnitude) in magnitude.iter().enumerate().take(num_bins) {
            let t = bin_t(bin_idx as f32);
            if t <= 0.0 || t >= 1.0 {
                continue;
            }

            let physical_x_coord = bounds.x + (bounds.w * t);
            let height = amplitude_to_height(magnitude, max_amplitude, bounds.h);

            // 动态调整高度
            let adjusted_height = height * normalized_db(rms_db); // 根据 RMS dB 调整高度

            // 绘制频谱条
            bars_path.move_to(
                physical_x_coord,
                bounds.y + bounds.h * (1.0 - adjusted_height),
            );
            bars_path.line_to(physical_x_coord, bounds.y + bounds.h);
        }
        canvas.stroke_path(&bars_path, &bars_paint);

        // 绘制横轴刻度尺
        let mut paint = vg::Paint::color(Color::black().into());
        let mut path = vg::Path::new();
        load_font_if_needed(canvas, &mut paint);
        paint.set_text_align(vg::Align::Center);
        paint.set_font_size(10.0);

        let num_ticks = 10; // 设置刻度数量
        let tick_interval = bounds.w / num_ticks as f32;

        for i in 0..=num_ticks {
            let x = bounds.x + i as f32 * tick_interval;
            let frequency = bin_frequency((i * (window_size / num_ticks) as usize) as f32);
            let frequency_str = format!("{:.0}", frequency);

            // 绘制刻度线
            path.move_to(x, 0.0);
            path.line_to(x, 10.0);
            
            // 绘制频率标签
            let _ = canvas.fill_text(x, 20.0, &frequency_str, &paint);
        }
        canvas.stroke_path(&path, &paint);
    }
}

// 下面全得放到helper类里

fn gain_to_db(gain: f32, _gain_factor: f32) -> f32 {
    let gain_factor = 1.0;
    // 不是固定的变量，怎么好看怎么来
    f32::max(gain * _gain_factor, MINUS_INFINITY_GAIN).log10() * 20.0
}

fn db_to_unclamped_t(db: f32, _height_factor: f32) -> f32 {
    let height_factor = 1.0;
    // 高度因子可以拉伸分贝值对应的绘图高度
    (db + 80.0) * _height_factor / 100.0
}

// 汉明窗？听着还以为是唐朝出来的呢
fn hamming_window(size: usize) -> Vec<f32> {
    let alpha = 0.54;
    let beta = 1.0 - alpha;
    (0..size)
        .map(|i| alpha - beta * (2.0 * std::f32::consts::PI * i as f32 / (size as f32 - 1.0)).cos())
        .collect()
}

// 德国窗
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size as f32 - 1.0)).cos()))
        .collect()
}

fn calculate_spectrum(samples: &[f32], window_size: usize) -> Vec<f32> {
    // 确保窗大小是2的幂，因为FFT通常对这种大小最有效
    let window_size = if window_size.is_power_of_two() {
        window_size
    } else {
        1 << window_size.trailing_zeros() // 找到小于等于window_size的最大2的幂
    };

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_size);
    let mut buffer: Vec<Complex<f32>> = Vec::with_capacity(window_size);

    // 应用汉明窗
    for i in 0..window_size {
        let alpha = 0.54; // 汉明窗系数
        let beta = 1.0 - alpha;
        let window = alpha
            - beta * (2.0 * std::f32::consts::PI * i as f32 / (window_size as f32 - 1.0)).cos();
        // 如果索引超出样本数组，则使用0填充
        let sample = if i < samples.len() { samples[i] } else { 0.0 };
        buffer.push(Complex::new(sample * window, 0.0));
    }

    fft.process(&mut buffer);

    // 只返回正频率部分的结果
    buffer
        .iter()
        .take(window_size / 2 + 1)
        .map(|c| c.norm())
        .collect()
}

fn smooth_spectrum_elder(magnitude: &mut [f32], window_size: usize) {
    let mut sum = 0.0;
    let mut smoothed_magnitude = Vec::with_capacity(magnitude.len());

    // 初始化滑动窗口
    for i in 0..window_size {
        sum += magnitude[i];
    }
    smoothed_magnitude.push(sum / window_size as f32);

    // 应用滑动平均
    for i in window_size..magnitude.len() {
        sum += magnitude[i] - magnitude[i - window_size];
        smoothed_magnitude.push(sum / window_size as f32);
    }

    // 确保平滑后的向量长度与原始向量长度相同
    // 如果原始向量长度大于平滑后向量，则复制最后一个平滑值
    while smoothed_magnitude.len() < magnitude.len() {
        smoothed_magnitude.push(*smoothed_magnitude.last().unwrap_or(&0.0));
    }

    // 将平滑后的频谱复制回原始向量
    magnitude.copy_from_slice(&smoothed_magnitude);
}

fn smooth_spectrum(spectrum: &mut [f32], smoothing_factor: usize) {
    let len = spectrum.len();
    for i in 0..len {
        let start = if i >= smoothing_factor {
            i - smoothing_factor
        } else {
            0
        };
        let end = if i + smoothing_factor < len {
            i + smoothing_factor
        } else {
            len - 1
        };
        let mut sum = 0.0;
        let mut count = 0;
        for j in start..=end {
            sum += spectrum[j];
            count += 1;
        }
        spectrum[i] = sum / count as f32;
    }
}

fn amplitude_to_height(amplitude: f32, max_amplitude: f32, bound_height: f32) -> f32 {
    if max_amplitude == 0.0 {
        return 0.0;
    }

    let min_db = -77.0; // 最低分贝
    let max_db = 0.0; // 最高分贝
    let amplitude_db = 20.0 * (amplitude / max_amplitude).log10();
    let amplitude_db = amplitude_db.clamp(min_db, max_db); // 处理超出范围的情况

    let db_range = max_db - min_db;
    let normalized_db = (amplitude_db - min_db) / db_range;

    // 使用对数变换来扩大范围
    let adjusted_db = normalized_db.powf(1.5); // 增加对数缩放因子

    adjusted_db.clamp(0.0, 1.0) * bound_height * 0.015 // 映射到高度
}

fn load_font_if_needed(canvas: &mut Canvas, paint: &mut vg::Paint) {
    FONT_LOADED.call_once(|| {
        let font_id = canvas.add_font("assets/Roboto-Regular.ttf").unwrap();
        paint.set_font(&[font_id]);
    });
}

fn calculate_rms(buffer: &[Complex<f32>]) -> f32 {
    let sum_of_squares: f32 = buffer.iter().map(|c| c.norm_sqr()).sum();
    let rms = (sum_of_squares / buffer.len() as f32).sqrt();
    rms
}

fn rms_to_db(rms: f32) -> f32 {
    if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        -77.0 // 处理静音或非常小的值
    }
}
