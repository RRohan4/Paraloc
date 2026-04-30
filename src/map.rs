pub struct Map {
    data: Vec<u8>,
    width: usize,
    height: usize,
}

impl Map {
    // Load a PNG from disk using the image crate, convert to grayscale,
    // treat any non-zero pixel as a wall, store as flat Vec<u8>
    pub fn from_path(path: &str) -> Self {
        let img = image::open(path).unwrap().to_luma8();
        let width = img.width() as usize;
        let height = img.height() as usize;
        let data = img.into_raw();
        
        Self::from_raw(data, width, height)

    }
    
    pub fn from_raw(data: Vec<u8>, width: usize, height: usize) -> Self {
        Self {data, width, height}
    }

    // Convert float world coords to tile indices (floor)
    pub fn is_wall(&self, x: f32, y: f32) ->bool {
        let tx = x as usize;
        let ty = y as usize;

        if tx >= self.width || ty >= self.height {
            return true;
        } 

        self.data[ty * self.width + tx] != 0
    }
    // dimensions(&self) -> (usize, usize)
    // Return (width, height)

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_map() -> Map {
        // 2x2: top-left is wall, rest open
        // W .
        // . .
        Map::from_raw(vec![1, 0, 0, 0], 2, 2)
    }

    #[test]
    fn wall_cell_is_detected() {
        assert!(simple_map().is_wall(0.0, 0.0));
    }

    #[test]
    fn open_cell_is_not_wall() {
        assert!(!simple_map().is_wall(1.0, 0.0));
    }

    #[test]
    fn out_of_bounds_is_treated_as_wall() {
        assert!(simple_map().is_wall(5.0, 5.0));
        assert!(simple_map().is_wall(-1.0, 0.0));
    }

    #[test]
    fn dimensions_are_correct() {
        assert_eq!(simple_map().dimensions(), (2, 2));
    }
}
