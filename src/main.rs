use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
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
            let downsampled_data: Vec<f32> = data.iter().step_by(10).copied().collect();
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

            SpectrumView::new(cx, AppData::waveform, sample_rate)
                .size(Pixels(800.0))
                .height(Pixels(150.0));
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
    sample_rate: u32,
}

impl<L> SpectrumView<L>
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

impl<L: Lens<Target = Arc<Mutex<Vec<f32>>>>> View for SpectrumView<L> {
    fn draw(&self, cx: &mut DrawContext, canvas: &Canvas) {
        let waveform = self.input_data.get(cx).lock().unwrap().clone();
        let bounds = cx.bounds();

        // 时域转频域（含-120dB下限）
        let spectrum = utils::spectrum::time_to_frequency(&waveform, -120.0);

        // 斜率优化 (-6dB/oct)
        let optimized =
            utils::spectrum::optimize_slope(&spectrum, self.sample_rate as f32, -6.0, 1000.0);

        // 生成三段式对数频率轴
        let (new_freqs, new_spectrum) = utils::spectrum::optimize_xaxis(
            &optimized,
            self.sample_rate as f32,
            100.0,  // 低频分界
            1000.0, // 中频分界
            100,    // 低频点数
            200,    // 中频点数
            50,     // 高频点数
        );

        // 空数据检查
        if new_freqs.is_empty() || new_spectrum.is_empty() {
            return;
        }

        // 坐标映射参数计算
        let min_freq = new_freqs[0].max(1e-6); // 防零保护
        let max_freq = *new_freqs.last().unwrap();
        let ln_range = max_freq.ln() - min_freq.ln();
        let db_min = -120.0; // 与time_to_frequency的floor保持一致
        let db_max = 0.0;

        // 创建频谱路径
        let mut path = vg::Path::new();
        let mut last_point: Option<vg::Point> = None;

        for (freq, db) in new_freqs.iter().zip(new_spectrum.iter()) {
            // 对数频率映射到X轴
            let x_progress = (freq.ln() - min_freq.ln()) / ln_range;
            let x = bounds.x + x_progress * bounds.w;

            // 分贝值映射到Y轴（上下留5%边距）
            let y_height = (db.clamp(db_min, db_max) - db_min) / (db_max - db_min);
            let y = bounds.y + bounds.h * 0.95 - y_height * bounds.h * 0.9;

            let point = vg::Point::new(x, y);

            if let Some(last) = last_point {
                // 添加贝塞尔曲线控制点实现平滑
                let ctrl_x = (last.x + point.x) / 2.0;
                path.quad_to(vg::Point::new(ctrl_x, last.y), point);
            } else {
                path.move_to(point);
            }
            last_point = Some(point);
        }

        // 配置画笔属性
        let mut paint = vg::Paint::default();
        paint.set_style(vg::PaintStyle::Stroke);
        // paint.set_color(vg::Color::from_rgb(0.0, 0.8, 0.2)); // 荧光绿

        let mut path = vg::Path::new();

        spectrum.iter().enumerate().for_each(|(i, db)| {

        });

        // 执行绘制
        canvas.draw_path(&path, &paint);

        // 可选：添加背景网格和刻度（此处需要额外实现）
        // self.draw_grid(cx, canvas, min_freq, max_freq);
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
