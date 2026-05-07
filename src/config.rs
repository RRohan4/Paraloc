use serde::Deserialize;

#[derive(Deserialize)]
#[serde(default)]
pub struct Config {
    pub particle_count: usize,
    pub translation_sigma: f32,
    pub rotation_sigma: f32,
    pub sensor_sigma: f32,
    pub score_sensor_sigma: f32,
    pub map_path: String,
    pub n_rays: usize,
    pub robot_spawn_x: f32,
    pub robot_spawn_y: f32,
    pub map_count: usize,
    pub map_size: usize,
    pub n_score_particles: usize,
    pub n_score_orientations: usize,
    pub init_radius_tiles: usize,
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            particle_count: 500,
            translation_sigma: 0.02,
            rotation_sigma: 0.02,
            sensor_sigma: 1.2,
            score_sensor_sigma: 15.0,
            map_path: "map.png".to_string(),
            n_rays: 36,
            robot_spawn_x: 18.5,
            robot_spawn_y: 14.5,
            map_count: 200,
            map_size: 128,
            n_score_particles: 200,
            n_score_orientations: 16,
            init_radius_tiles: 10,
            output_dir: "dataset_out".to_string(),
        }
    }
}

impl Config {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        let config = toml::from_str(&text)?;
        Ok(config)
    }

    pub fn load_or_default(path: &str) -> Self {
        Self::from_file(path).unwrap_or_default()
    }
}
