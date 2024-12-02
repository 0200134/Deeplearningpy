extern crate linfa;
extern crate linfa_preprocessing;
extern crate ndarray;

use linfa::prelude::*;
use linfa::DatasetView;
use linfa::metrics::{confusion_matrix, ClassificationSummary};
use linfa_knn::KNearestNeighbors;
use linfa_preprocessing::StandardScaler;
use ndarray::array;

fn main() {
    // Load the iris dataset
    let dataset = linfa_datasets::iris();
    let (train, valid) = dataset
        .shuffle(42)
        .split_with_ratio(0.8);

    // Standardize the dataset
    let scaler = StandardScaler::default();
    let (train, scaler) = scaler.fit_transform(train);
    let valid = scaler.transform(valid);

    // Create and train the k-nearest neighbors classifier
    let model = KNearestNeighbors::params().k(3).fit(&train).unwrap();

    // Make predictions
    let pred = model.predict(&valid);

    // Evaluate the model
    let confusion_matrix = confusion_matrix(&valid, &pred);
    println!("Confusion Matrix:\n{:?}", confusion_matrix);

    let summary = ClassificationSummary::from_confusion_matrix(&confusion_matrix);
    println!("\nClassification Report:\n{:?}", summary);
}
