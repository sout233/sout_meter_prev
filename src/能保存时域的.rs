    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let mut data = self.data.get(cx).lock().unwrap().clone();
        let bounds = cx.bounds();
    
        // 计算每个像素对应的样本数
        let sample_rate = 48000.0;
        let time_per_pixel = 1.0 / sample_rate; // 每个像素代表的时间
        let samples_per_pixel = sample_rate / bounds.w; // 每个像素对应的样本数
    
        // 确保显示区域内有足够的数据样本
        let max_samples = (bounds.w as f32 * samples_per_pixel).ceil() as usize;
        if data.len() > max_samples {
            data.drain(..data.len() - max_samples);
        }
    
        let mut path = vg::Path::new();
        let step = bounds.w / (data.len() - 1) as f32;
    
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