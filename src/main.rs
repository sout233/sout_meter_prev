use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use pitch_detection::detector::mcleod::McLeodDetector;
use pitch_detection::detector::PitchDetector;
use std::sync::{Arc, Mutex};
use std::thread;
use vizia::prelude::*;
use vizia::vg;

mod utils;

const SIZE: usize = 1024;
const PADDING: usize = SIZE / 2;
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

        let mut signal: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        let mut period_samples = bounds.w as usize; // 随窗口而变

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

            // signal.resize(SIZE, 0.0);
            // let mut detector = McLeodDetector::new(SIZE, PADDING);

            // let sample_rate = self.sample_rate.try_into().unwrap_or(48000);
            // let pitch = detector.get_pitch(
            //     &signal,
            //     sample_rate,
            //     POWER_THRESHOLD,
            //     CLARITY_THRESHOLD,
            // );

            // // 如果检测到有效基频，计算周期样本数
            // if let Some(pitch) = pitch {
            //     let calculated_period = (sample_rate as f64 / pitch.frequency) as usize;
            //     period_samples = calculated_period.max(2); // 两点成线

            //     println!("Base Frequency: {:.2}Hz | Period Samples: {} | Clarity: {:.2}",
            //         pitch.frequency,
            //         period_samples,
            //         pitch.clarity
            //     );
            // }
        }

        // // 截取或扩展数据到单个周期长度
        // if data.len() > period_samples {
        //     // 保留最新一个周期的数据
        //     data.drain(..data.len() - period_samples);
        // } else if data.len() < period_samples {
        //     // 补零，用resize或许也行但是我懒得改了
        //     data.extend(vec![0.0; period_samples - data.len()]);
        // }

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
