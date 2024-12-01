import Foundation
import CoreML

// A class to handle AI model operations
class AIModel {
    let model: MLModel

    // Initialization with the model name
    init(modelName: String) {
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            fatalError("Model not found")
        }
        do {
            model = try MLModel(contentsOf: modelURL)
        } catch {
            fatalError("Failed to load model: \(error)")
        }
    }

    // Prediction function using provided input
    func predict(input: MLFeatureProvider) -> MLFeatureProvider? {
        do {
            let prediction = try model.prediction(from: input)
            return prediction
        } catch {
            print("Prediction failed: \(error)")
            return nil
        }
    }
}

// Example usage of the AIModel class
if let input = try? MLMultiArray(shape: [1], dataType: .double) {
    input[0] = 0.5
    let featureProvider = try? MLDictionaryFeatureProvider(dictionary: ["input": input])
    
    if let featureProvider = featureProvider {
        let aiModel = AIModel(modelName: "YourModelName")
        if let prediction = aiModel.predict(input: featureProvider) {
            // Process the prediction
            print("Prediction: \(prediction)")
        }
    }
}
