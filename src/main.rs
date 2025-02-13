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

        WaveformView::new(cx, AppData::waveform, sample_rate)
            .size(Pixels(800.0))
            .height(Pixels(150.0));

        let timer = cx.add_timer(Duration::from_millis(33), None, |cx, _| {
            cx.emit(AppEvent::UpdateWaveform);
        });

        cx.start_timer(timer);
    })
    .inner_size((800, 300))
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