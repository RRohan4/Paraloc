use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub particle_count: usize,
    pub translation_sigma: f32,
    pub rotation_sigma: f32,
    pub sensor_sigma: f32,
    pub map_path: String,
    pub n_rays: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            particle_count: 500,
            translation_sigma: 0.02,
            rotation_sigma: 0.02,
            sensor_sigma: 0.2,
            map_path: "map.png".to_string(),
            n_rays: 36,
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
