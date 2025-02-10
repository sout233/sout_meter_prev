use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use vizia::prelude::*;
use vizia::vg;

fn main() -> Result<(), ApplicationError> {
    // 用于存储原始采样数据
    let audio_buffer = Arc::new(Mutex::new(Vec::new()));
    // 用于冻结之后的单周期数据，初始为空
    let frozen_cycle = Arc::new(Mutex::new(None));

    let buffer_clone = audio_buffer.clone();
    // 音频采集线程（注意：此处依然用的是 default_output_device，
    // 若需要采集输入，请使用 default_input_device）
    thread::spawn(move || {
        let host = cpal::default_host();
        let err_fn = |err| eprintln!("an error occurred on the audio stream: {}", err);
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

        // 为了减少数据量，这里对采样数据做下采样（比如每 10 个点取一个）
        let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buffer = buffer_clone.lock().unwrap();
            let downsampled_data: Vec<f32> = data.iter().step_by(10).copied().collect();
            buffer.extend(downsampled_data);
        };

        let stream = match sample_format {
            _ => device.build_input_stream(&config, input_data_fn, err_fn, None),
        }
        .unwrap();

        stream.play().unwrap();
        loop {
            thread::sleep(Duration::from_millis(1000));
        }
    });

    Application::new(move |cx| {
        AppData {
            waveform: audio_buffer.clone(),
            frozen_cycle: frozen_cycle.clone(),
        }
        .build(cx);

        // 创建 WaveformView，将 audio_buffer 与 frozen_cycle 两个状态传进去
        WaveformView::new(cx, AppData::waveform, AppData::frozen_cycle)
            .size(Pixels(800.0))
            .height(Pixels(150.0));

        // 这里依然使用定时器更新界面，但由于 frozen_cycle 一旦设置后就不会改变，
        // 所以界面渲染的内容将保持固定
        let timer = cx.add_timer(Duration::from_millis(33), None, |cx, _| {
            cx.emit(AppEvent::UpdateWaveform);
        });
        cx.start_timer(timer);
    })
    .inner_size((800, 300))
    .title("固定单周期波形")
    .run()
}

#[derive(Lens, Clone)]
struct AppData {
    // 原始采样数据
    waveform: Arc<Mutex<Vec<f32>>>,
    // 冻结的单周期数据（None 表示还没有检测到完整周期）
    frozen_cycle: Arc<Mutex<Option<Vec<f32>>>>,
}

impl Model for AppData {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|app_event, _| match app_event {
            AppEvent::UpdateWaveform => {
                // 这里只是触发重绘
                cx.needs_redraw();
            }
        });
    }
}

pub enum AppEvent {
    UpdateWaveform,
}

/// 使用 YIN 算法检测周期，返回该周期的采样点数
fn detect_period_yin(samples: &[f32]) -> Option<usize> {
    let n = samples.len();
    if n < 2 {
        return None;
    }
    // 限制最大可能周期为样本数量的一半
    let tau_max = n / 2;
    let mut d = vec![0.0; tau_max];
    // 计算差分函数，从 tau = 1 开始
    for tau in 1..tau_max {
        let mut sum = 0.0;
        for j in 0..(n - tau) {
            let diff = samples[j] - samples[j + tau];
            sum += diff * diff;
        }
        d[tau] = sum;
    }
    // 计算累计均值归一化差分函数
    let mut d_norm = vec![0.0; tau_max];
    d_norm[0] = 1.0;
    let mut running_sum = 0.0;
    for tau in 1..tau_max {
        running_sum += d[tau];
        d_norm[tau] = d[tau] * (tau as f32) / running_sum;
    }
    // 阈值通常设在 0.1～0.2 之间
    let threshold = 0.1;
    for tau in 1..tau_max {
        if d_norm[tau] < threshold {
            return Some(tau);
        }
    }
    None
}

/// WaveformView 同时获取原始采样数据和冻结周期数据
pub struct WaveformView<L, F>
where
    L: Lens<Target = Arc<Mutex<Vec<f32>>>>,
    F: Lens<Target = Arc<Mutex<Option<Vec<f32>>>>>,
{
    data: L,
    frozen: F,
}

impl<L, F> WaveformView<L, F>
where
    L: Lens<Target = Arc<Mutex<Vec<f32>>>>,
    F: Lens<Target = Arc<Mutex<Option<Vec<f32>>>>>,
{
    pub fn new(cx: &mut Context, data: L, frozen: F) -> Handle<Self> {
        Self { data, frozen }
            .build(cx, |_| {})
            .bind(data, |mut handle, _| handle.needs_redraw())
            .bind(frozen, |mut handle, _| handle.needs_redraw())
    }
}

impl<L, F> View for WaveformView<L, F>
where
    L: Lens<Target = Arc<Mutex<Vec<f32>>>>,
    F: Lens<Target = Arc<Mutex<Option<Vec<f32>>>>>,
{
    fn draw(&self, cx: &mut DrawContext, canvas: &Canvas) {
        // 优先检查是否已有冻结的周期数据
        let binding = self.frozen.get(cx);
        let mut frozen_lock = binding.lock().unwrap();
        if let Some(ref cycle) = *frozen_lock {
            draw_cycle(cx, canvas, cycle);
            return;
        }

        // 如果还没有冻结周期，则从原始数据中检测周期
        let mut data = self.data.get(cx).lock().unwrap().clone();
        // 为了防止数据过多，每次只保留最近的 2048 个采样点
        if data.len() > 2048 {
            data = data.split_off(data.len() - 2048);
        }
        // 使用 YIN 算法检测周期
        if let Some(period) = detect_period_yin(&data) {
            if period < data.len() {
                // 提取出最后一个完整周期
                let cycle = data.split_off(data.len() - period);
                // 冻结数据，后续均使用此数据
                *frozen_lock = Some(cycle.clone());
                drop(frozen_lock);
                draw_cycle(cx, canvas, &cycle);
                return;
            }
        }
        // 如果没有检测到周期，则暂时不绘制内容
    }
}

/// 辅助函数：根据周期数据绘制波形
fn draw_cycle(cx: &mut DrawContext, canvas: &Canvas, cycle: &[f32]) {
    let bounds = cx.bounds();
    if cycle.len() < 2 {
        return;
    }
    // 将周期数据横向拉伸铺满整个绘图区域
    let step = bounds.w / (cycle.len() - 1) as f32;
    let mut path = vg::Path::new();
    for (i, &value) in cycle.iter().enumerate() {
        let x = bounds.x + i as f32 * step;
        let y = bounds.y + bounds.h * (1.0 - value);
        let point = vg::Point::new(x, y);
        if i == 0 {
            path.move_to(point);
        } else {
            path.line_to(point);
        }
    }
    let mut paint = vg::Paint::default();
    paint.set_style(vg::PaintStyle::Stroke);
    canvas.draw_path(&path, &paint);
}
