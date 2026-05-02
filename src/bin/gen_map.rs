use image::{GrayImage, Luma};

const W: u32 = 40;
const H: u32 = 40;
const WALL: Luma<u8> = Luma([0]);
const OPEN: Luma<u8> = Luma([255]);

fn put_h(img: &mut GrayImage, y: u32, x0: u32, x1: u32) {
    for x in x0..x1 {
        img.put_pixel(x, y, WALL);
    }
}

fn put_v(img: &mut GrayImage, x: u32, y0: u32, y1: u32) {
    for y in y0..y1 {
        img.put_pixel(x, y, WALL);
    }
}

fn main() {
    let mut img = GrayImage::from_pixel(W, H, OPEN);

    for x in 0..W {
        img.put_pixel(x, 0, WALL);
        img.put_pixel(x, H - 1, WALL);
    }
    for y in 0..H {
        img.put_pixel(0, y, WALL);
        img.put_pixel(W - 1, y, WALL);
    }

    put_v(&mut img, 12, 1, 14);
    put_h(&mut img, 14, 1, 13);
    img.put_pixel(12, 7, OPEN);
    img.put_pixel(12, 8, OPEN);

    put_h(&mut img, 5, 18, 38);
    put_v(&mut img, 28, 5, 14);
    img.put_pixel(28, 9, OPEN);
    img.put_pixel(28, 10, OPEN);
    img.put_pixel(33, 5, OPEN);
    img.put_pixel(34, 5, OPEN);

    put_h(&mut img, 22, 1, 18);
    img.put_pixel(8, 22, OPEN);
    img.put_pixel(9, 22, OPEN);

    put_v(&mut img, 22, 22, 38);
    img.put_pixel(22, 28, OPEN);
    img.put_pixel(22, 29, OPEN);

    put_h(&mut img, 30, 25, 38);
    img.put_pixel(32, 30, OPEN);
    img.put_pixel(33, 30, OPEN);

    for i in 0..6u32 {
        img.put_pixel(16 + i, 14 + i, WALL);
    }

    img.put_pixel(6, 30, WALL);
    img.put_pixel(7, 30, WALL);
    img.put_pixel(6, 31, WALL);
    img.put_pixel(7, 31, WALL);

    img.put_pixel(34, 24, WALL);
    img.put_pixel(35, 24, WALL);
    img.put_pixel(34, 25, WALL);

    img.put_pixel(15, 35, WALL);
    img.put_pixel(16, 35, WALL);
    img.put_pixel(15, 36, WALL);

    img.save("map.png").unwrap();
    println!("wrote map.png ({}x{})", W, H);
}
