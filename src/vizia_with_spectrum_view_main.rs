use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::sync::{Arc, Mutex};
use std::thread;
use vizia::prelude::*;
use vizia::vg;

fn main() -> Result<(), ApplicationError> {
    let audio_buffer = Arc::new(Mutex::new(Vec::new()));

    // 音频输入线程？启动！
    let buffer_clone = audio_buffer.clone();
    thread::spawn(move || {
        let host = cpal::default_host();

        let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);

        let device = host.default_input_device().unwrap();

        let mut supported_configs_range = device.supported_input_configs().unwrap();
        let supported_config = supported_configs_range
            .next()
            .unwrap()
            .with_max_sample_rate();

        let sample_format = supported_config.sample_format();

        let config = supported_config.into();

        let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buffer = buffer_clone.lock().unwrap();
            // Downsample the audio data by a factor of 10
            let downsampled_data: Vec<f32> = data.iter().step_by(10).copied().collect();
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

        WaveformView::new(cx, AppData::waveform)
            .size(Pixels(800.0))
            .height(Pixels(150.0));

        SpectrumView::new(cx, AppData::waveform)
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
    data: L,
}

impl<L: Lens<Target = Arc<Mutex<Vec<f32>>>>> WaveformView<L> {
    pub fn new(cx: &mut Context, data: L) -> Handle<Self> {
        Self { data }
            .build(cx, |_| {})
            .bind(data, |mut handle, _| handle.needs_redraw())
    }
}

impl<L: Lens<Target = Arc<Mutex<Vec<f32>>>>> View for WaveformView<L> {
    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let mut data = self.data.get(cx).lock().unwrap().clone();
        let bounds = cx.bounds();
        let max_samples = (bounds.w as usize).min(data.len());

        // Ensure we only render the current window size worth of data
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

// SpectrumView

pub struct SpectrumView<L: Lens<Target = Arc<Mutex<Vec<f32>>>>> {
    data: L,
}

impl<L: Lens<Target = Arc<Mutex<Vec<f32>>>>> SpectrumView<L> {
    pub fn new(cx: &mut Context, data: L) -> Handle<Self> {
        Self { data }
            .build(cx, |_| {})
            .bind(data, |mut handle, _| handle.needs_redraw())
    }
}

impl<L: Lens<Target = Arc<Mutex<Vec<f32>>>>> View for SpectrumView<L> {
    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let pre_data = self.data.get(cx).lock().unwrap().clone();
        let mut data = calculate_spectrum(&pre_data);
        let bounds = cx.bounds();
        let max_samples = (bounds.w as usize).min(data.len());

        // Ensure we only render the current window size worth of data
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

        let paint = vg::Paint::color(Color::blueviolet().into()).with_line_width(2.0);
        canvas.stroke_path(&path, &paint);
    }
}

pub fn calculate_spectrum(samples: &[f32]) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(samples.len());
    let mut buffer: Vec<Complex<f32>> = samples
        .iter()
        .map(|&s| Complex { re: s, im: 0.0 })
        .collect();
    fft.process(&mut buffer);

    buffer.iter().map(|c| c.norm()).collect()
}
