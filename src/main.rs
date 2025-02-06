#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints};
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::btree_map::Values;
use std::sync::{Arc, Mutex};
use std::thread;

fn main() -> eframe::Result {
    let audio_buffer = Arc::new(Mutex::new(Vec::new()));
    let spectrum_buffer = Arc::new(Mutex::new(Vec::new()));

    let buffer_clone = audio_buffer.clone();
    let spectrum_clone = spectrum_buffer.clone();

    // 音频输入线程？启动！
    thread::spawn(move || {
        let host = cpal::default_host();
        let device = host.default_output_device().unwrap();

        let config = device.default_output_config().unwrap();

        let sample_rate = config.sample_rate().0 as f32;
        let channels = config.channels() as usize;

        let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);

        let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buffer = buffer_clone.lock().unwrap();
            buffer.extend_from_slice(data);

            let window_size = 1024; // FFT窗口大小
            let hop_size = 512; // FFT窗口跳过的样本数

            if buffer.len() >= window_size {
                let mut fft_buffer: Vec<Complex<f32>> = buffer
                    .drain(..window_size)
                    .map(|x| Complex { re: x, im: 0.0 })
                    .collect();

                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(window_size);
                fft.process(&mut fft_buffer);

                let spectrum = fft_buffer.iter().map(|c| c.norm()).collect::<Vec<f32>>();
                let mut spectrum_guard = spectrum_clone.lock().unwrap();
                *spectrum_guard = spectrum;
            }
        };

        let stream = device
            .build_input_stream(&config.into(), input_data_fn, err_fn, None)
            .unwrap();
        stream.play().unwrap();

        loop {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([700.0, 400.0]),
        ..Default::default()
    };
    let app = MyApp::new(spectrum_buffer);
    eframe::run_native("My egui App", options, Box::new(|cc| Ok(Box::new(app))))
}

struct MyApp {
    spectrum_buffer: Arc<Mutex<Vec<f32>>>,
}

impl MyApp {
    fn new(spectrum_buffer: Arc<Mutex<Vec<f32>>>) -> MyApp {
        Self { spectrum_buffer }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Audio Spectrum");

            let spectrum = self.spectrum_buffer.lock().unwrap();
            let spectrum_data: Vec<f32> = spectrum.iter().cloned().collect();
            // 确保频谱数据长度大于0
            if spectrum_data.len() == 0 {
                return; // 如果没有数据，则不进行绘制
            }
            // 限制频段范围，例如只显示20Hz到20kHz
            let min_freq = 20.0;
            let max_freq = 20000.0;
            let nyquist_freq = 48000.0 / 2.0; // 奈奎斯特频率
            let min_index = ((min_freq / nyquist_freq) * spectrum_data.len() as f32) as usize;
            let max_index = ((max_freq / nyquist_freq) * spectrum_data.len() as f32) as usize;
            if min_index >= max_index
                || min_index >= spectrum_data.len()
                || max_index > spectrum_data.len()
            {
                return; // 如果索引不正确，则不进行绘制
            }

            // 获取频谱数据的前半部分
            let half_spectrum_data: Vec<f32> = spectrum_data
                .iter()
                .take(spectrum_data.len() / 2)
                .cloned()
                .collect();

            let num_bars = 30;
            let bar_width = if num_bars > 0 {
                (max_index - min_index + 1) / num_bars
            } else {
                return; // 如果 num_bars 为0，则不进行绘制
            };
            let mut bars: Vec<Bar> = Vec::new();

            // 计算每个条形的平均值
            for i in 0..num_bars {
                let start = i * bar_width;
                // 确保 end 不超出范围
                let end = (start + bar_width).min(half_spectrum_data.len());
                let bar_data: Vec<f32> = half_spectrum_data[start..end].to_vec();
                let bar_avg = bar_data.iter().sum::<f32>() / bar_data.len() as f32;
                // 转换为对数响应级别（可选）
                let log_bar_avg = bar_avg.log10().max(0.0); // 避免负数
                bars.push(Bar::new(i as f64, log_bar_avg as f64));
            }
            // 创建条形图
            let bar_chart = BarChart::new(bars)
                .name("Spectrum")
                .color(egui::Color32::from_rgb(0, 255, 0)); // 绿色

            // 显示条形图
            Plot::new("audio_spectrum")
                .view_aspect(2.0)
                .show_x(false)
                .show_y(false)
                .show(ui, |plot_ui| plot_ui.bar_chart(bar_chart));
        });

        ctx.request_repaint();
        std::thread::sleep(std::time::Duration::from_secs_f64(1.0 / 60.0));
    }
}
