extern crate linfa;
extern crate linfa_preprocessing;
extern crate ndarray;

use linfa::prelude::*;
use linfa::metrics::{confusion_matrix, ClassificationSummary, cross_validate};
use linfa_knn::KNearestNeighbors;
use linfa_preprocessing::StandardScaler;
use ndarray::array;
use std::fs::File;
use std::io::Write;
use log::{info, LevelFilter};
use simplelog::{CombinedLogger, Config, TermLogger, WriteLogger};

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
    let model = KNearestNeighbors::params().k(best_k).fit(&train).unwrap();

    // Make predictions
    let pred = model.predict(&valid);

    // Evaluate the model
    let confusion_matrix = confusion_matrix(&valid, &pred);
    println!("Confusion Matrix:\n{:?}", confusion_matrix);

    let summary = ClassificationSummary::from_confusion_matrix(&confusion_matrix);
    println!("\nClassification Report:\n{:?}", summary);

    info!("Confusion Matrix: {:?}", confusion_matrix);
    info!("Classification Report: {:?}", summary);
}
