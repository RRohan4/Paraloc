use serde::Deserialize;

#[derive(Deserialize)]
#[serde(default)]
pub struct Config {
    pub score_sensor_sigma: f32,
    pub n_rays: usize,
    pub map_size: usize,
    pub n_score_particles: usize,
    pub n_score_orientations: usize,
    pub init_radius_tiles: usize,
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            score_sensor_sigma: 15.0,
            n_rays: 36,
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
        Ok(toml::from_str(&text)?)
    }

    pub fn load_or_default(path: &str) -> Self {
        Self::from_file(path).unwrap_or_default()
    }
}
