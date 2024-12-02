extern crate linfa;
extern crate linfa_preprocessing;
extern crate ndarray;
extern crate serde;
extern crate serde_json;

use linfa::prelude::*;
use linfa::metrics::{confusion_matrix, ClassificationSummary, cross_validate};
use linfa_knn::KNearestNeighbors;
use linfa_preprocessing::StandardScaler;
use ndarray::array;
use std::fs::File;
use std::io::{Write, Read};
use log::{info, LevelFilter};
use simplelog::{CombinedLogger, Config, TermLogger, WriteLogger};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct ModelWrapper {
    scaler: StandardScaler,
    knn: KNearestNeighbors<f32>,
}

impl ModelWrapper {
    fn new(scaler: StandardScaler, knn: KNearestNeighbors<f32>) -> Self {
        Self { scaler, knn }
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let serialized = serde_json::to_string(self).unwrap();
        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    fn load(path: &str) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let deserialized: Self = serde_json::from_str(&contents).unwrap();
        Ok(deserialized)
    }
}

fn main() {
    // Initialize logging
    CombinedLogger::init(vec![
        TermLogger::new(LevelFilter::Info, Config::default(), simplelog::TerminalMode::Mixed).unwrap(),
        WriteLogger::new(LevelFilter::Info, Config::default(), File::create("log.txt").unwrap()),
    ]).unwrap();

    // Load the iris dataset
    let dataset = linfa_datasets::iris();
    let (train, valid) = dataset
        .shuffle(42)
        .split_with_ratio(0.8);

    // Standardize the dataset
    let scaler = StandardScaler::default();
    let (train, scaler) = scaler.fit_transform(train);
    let valid = scaler.transform(valid);

    // Hyperparameter tuning: testing different values of k
    let mut best_k = 1;
    let mut best_score = 0.0;

    for k in 1..20 {
        let model = KNearestNeighbors::params().k(k).fit(&train).unwrap();
        let scores = cross_validate(5, &train, |train, valid| {
            let model = model.clone();
            model.fit(train).unwrap().predict(valid)
        }).unwrap();
        let mean_score = scores.mean_accuracy();
        
        if mean_score > best_score {
            best_k = k;
            best_score = mean_score;
        }
    }
    
    info!("Best k: {}, Best cross-validated accuracy: {}", best_k, best_score);

    // Train the final model with the best hyperparameter
    let knn_model = KNearestNeighbors::params().k(best_k).fit(&train).unwrap();

    // Save the trained model
    let model_wrapper = ModelWrapper::new(scaler, knn_model);
    model_wrapper.save("model.json").unwrap();

    // Load the trained model
    let loaded_model = ModelWrapper::load("model.json").unwrap();

    // Make predictions using the loaded model
    let pred = loaded_model.knn.predict(&valid);

    // Evaluate the model
    let confusion_matrix = confusion_matrix(&valid, &pred);
    println!("Confusion Matrix:\n{:?}", confusion_matrix);

    let summary = ClassificationSummary::from_confusion_matrix(&confusion_matrix);
    println!("\nClassification Report:\n{:?}", summary);

    info!("Confusion Matrix: {:?}", confusion_matrix);
    info!("Classification Report: {:?}", summary);
}
