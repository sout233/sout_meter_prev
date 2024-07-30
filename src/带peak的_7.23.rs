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
            cx.emit(WindowEvent::DragWindow);
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
        average*=10000.0;
        let current_sample = *data.last().unwrap_or(&0.0);
        if average > 0.0 {
            // println!("current_sample: {}", current_sample);
            // println!("average: {}", average);
        }
        // 将样本值转换为dB值
        let db_value = sample_to_db(current_sample) + 0.0;
        // 映射dB值到Y轴，假设dB值在[-60, 0]之间
        let y = bounds.y + bounds.h * (1.0 - (db_value + 60.0) / 60.0)+10.0;

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
        let mut paint = vg::Paint::color(Color::black().into());

        // 感谢geom3trik的指点
        load_font_if_needed(canvas, &mut paint);

        paint.set_text_align(vg::Align::Center);
        paint.set_font_size(10.0);

        // TODO: 总不能只渲染一个test吧
        let _ = canvas.fill_text(100.0, 20.0, "sout meter test", &paint);

        let data = self.data.get(cx).lock().unwrap().clone();
        let spectrum = self.spectrum.get(cx).lock().unwrap().clone();
        let bounds = cx.bounds();

        let gain_factor = 10.0; // 增益因子，没啥用
        let height_factor = 2.0; // 高度因子，也没啥用

        // 玄学优化，有总比没有强
        if data == spectrum {
            return;
        }

        // 窗函数很重要
        let window_size = 1024;
        if data.len() < window_size {
            return;
        }

        let window = hamming_window(window_size);
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
        // smooth_spectrum(&mut magnitude, 5);

        let max_magnitude = magnitude.iter().cloned().fold(0.0 / 0.0, f32::max);

        let line_width = cx.scale_factor() * 1.5;
        // let text_color: vg::Color = vg::Color::rgb(136,20,0);
        let text_color: vg::Color = vg::Color::black();
        let bars_paint = vg::Paint::color(text_color).with_line_width(line_width);
        let mut lighter_text_color = text_color;
        lighter_text_color.r = (lighter_text_color.r + 0.25) / 1.25;
        lighter_text_color.g = (lighter_text_color.g + 0.25) / 1.25;
        lighter_text_color.b = (lighter_text_color.b + 0.25) / 1.25;

        let nyquist_hz = 48000.0 / 2.0; // TODO: 这玩意应该dynamic的
        let bin_frequency = |bin_idx: f32| (bin_idx / (window_size as f32 / 2.0)) * nyquist_hz;
        let bin_t =
            |bin_idx: f32| (bin_frequency(bin_idx).ln() - LN_FREQ_RANGE_START_HZ) / LN_FREQ_RANGE;
        let magnitude_height = |magnitude: f32| {
            let magnitude_db = gain_to_db(magnitude / max_magnitude, gain_factor);
            db_to_unclamped_t(magnitude_db, height_factor).clamp(0.0, 2.0)
        };

        let mesh_start_delta_threshold = line_width + 0.5;
        let mut mesh_bin_start_idx = window_size / 2;
        let mut previous_physical_x_coord = bounds.x - 2.0;

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
            if physical_x_coord - previous_physical_x_coord < mesh_start_delta_threshold {
                mesh_bin_start_idx = bin_idx.saturating_sub(1);
                previous_physical_x_coord = physical_x_coord;
                break;
            }

            let physical_x_coord = bounds.x + (bounds.w * t);
            // let height = magnitude_height(magnitude / max_magnitude);
            let height = amplitude_to_height(magnitude, max_amplitude);

            // 给我待在底部！！！
            bars_path.move_to(physical_x_coord, bounds.y + bounds.h * (1.0 - height));
            bars_path.line_to(physical_x_coord, bounds.y + bounds.h);
        }
        canvas.stroke_path(&bars_path, &bars_paint);

        // 下面的是网格（？我也不清楚，毕竟是从nihplug那边拿来的

        let mut mesh_path = vg::Path::new();
        let mesh_start_x_coordiante = bounds.x + (bounds.w * bin_t(mesh_bin_start_idx as f32));
        let mesh_start_y_coordinate = bounds.y + bounds.h;

        mesh_path.move_to(mesh_start_x_coordiante, mesh_start_y_coordinate);
        for (bin_idx, &magnitude) in magnitude
            .iter()
            .enumerate()
            .take(window_size / 2)
            .skip(mesh_bin_start_idx)
        {
            let t = bin_t(bin_idx as f32);
            if t <= 0.0 || t >= 1.0 {
                continue;
            }

            let physical_x_coord = bounds.x + (bounds.w * t);
            previous_physical_x_coord = physical_x_coord;
            let height = magnitude_height(magnitude / max_magnitude);
            if height > 0.0 {
                mesh_path.line_to(
                    physical_x_coord,
                    bounds.y + (bounds.h * (1.0 - height) - (line_width / 2.0)).max(0.0),
                );
            } else {
                mesh_path.line_to(physical_x_coord, mesh_start_y_coordinate);
            }
        }

        mesh_path.line_to(previous_physical_x_coord, mesh_start_y_coordinate);
        mesh_path.close();

        let mesh_paint = vg::Paint::linear_gradient_stops(
            mesh_start_x_coordiante,
            0.0,
            previous_physical_x_coord,
            0.0,
            [
                (0.0, lighter_text_color),
                (0.707, text_color),
                (1.0, text_color),
            ],
        )
        .with_anti_alias(false);
        canvas.fill_path(&mesh_path, &mesh_paint);
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

pub fn calculate_spectrum(samples: &[f32], window_size: usize) -> Vec<f32> {
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

fn smooth_spectrum(magnitude: &mut [f32], window_size: usize) {
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

fn amplitude_to_height(amplitude: f32, max_amplitude: f32) -> f32 {
    // 使用对数函数来转换幅度到高度
    let min_db = -80.0;
    let max_db = 0.0;
    let db_range = max_db - min_db;
    let amplitude_db = 20.0 * (amplitude / max_amplitude).log10();
    let normalized_db = (amplitude_db - min_db) / db_range;
    normalized_db.clamp(0.0, 1.0) * 1.5
}

fn load_font_if_needed(canvas: &mut Canvas, paint: &mut vg::Paint) {
    FONT_LOADED.call_once(|| {
        let font_id = canvas.add_font("assets/Roboto-Regular.ttf").unwrap();
        paint.set_font(&[font_id]);
    });
}
