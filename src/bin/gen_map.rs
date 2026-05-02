use image::{GrayImage, Luma};

const W: u32 = 50;
const H: u32 = 50;
const WALL: Luma<u8> = Luma([0]);
const OPEN: Luma<u8> = Luma([255]);

fn fill(img: &mut GrayImage, x0: u32, y0: u32, x1: u32, y1: u32, p: Luma<u8>) {
    for y in y0..y1 {
        for x in x0..x1 {
            img.put_pixel(x, y, p);
        }
    }
}

fn main() {
    let mut img = GrayImage::from_pixel(W, H, WALL);

    let corridor_cols = [(5u32, 10u32), (22, 27), (39, 44)];
    for &(c0, c1) in &corridor_cols {
        fill(&mut img, c0, 1, c1, 25, OPEN);
    }

    for &(c0, c1) in &corridor_cols {
        let right_notch = c1 - 1;
        for y in 8..11 {
            img.put_pixel(right_notch, y, WALL);
        }
        let left_notch = c0;
        for y in 16..19 {
            img.put_pixel(left_notch, y, WALL);
        }
    }

    fill(&mut img, 1, 27, W - 1, H - 1, OPEN);

    for &(c0, c1) in &corridor_cols {
        fill(&mut img, c0, 25, c1, 27, OPEN);
    }

    for i in 0..14u32 {
        let x = 13 + i;
        let y = 30 + i;
        if x < W - 1 && y < H - 1 {
            img.put_pixel(x, y, WALL);
            img.put_pixel(x + 1, y, WALL);
        }
    }

    for x in 30..42 {
        img.put_pixel(x, 35, WALL);
    }
    for y in 35..45 {
        img.put_pixel(41, y, WALL);
    }

    fill(&mut img, 1, 38, 4, 45, WALL);
    img.put_pixel(4, 41, WALL);

    fill(&mut img, 8, 30, 11, 33, WALL);

    fill(&mut img, 44, 30, 47, 32, WALL);
    img.put_pixel(46, 32, WALL);

    for x in 0..W {
        img.put_pixel(x, 0, WALL);
        img.put_pixel(x, H - 1, WALL);
    }
    for y in 0..H {
        img.put_pixel(0, y, WALL);
        img.put_pixel(W - 1, y, WALL);
    }

    img.save("map.png").unwrap();
    println!("wrote map.png ({}x{})", W, H);
}
