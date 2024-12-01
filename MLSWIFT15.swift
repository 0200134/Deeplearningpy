import Foundation
import CoreML

class DeepLearningModel {
    let model: MLModel

    init() {
        guard let modelURL = Bundle.main.url(forResource: "DeepLearningModel", withExtension: "mlmodelc") else {
            fatalError("Model not found")
        }
        do {
            model = try MLModel(contentsOf: modelURL)
        } catch {
            fatalError("Failed to load model: \(error)")
        }
    }

    func predict(input: MLMultiArray) -> MLMultiArray? {
        let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["input": input])
        
        guard let inputProvider = inputProvider else {
            return nil
        }
        
        do {
            let prediction = try model.prediction(from: inputProvider)
            return prediction.featureValue(for: "output")?.multiArrayValue
        } catch {
            print("Prediction failed: \(error)")
            return nil
        }
    }
}

// Example usage
if let input = try? MLMultiArray(shape: [64], dataType: .double) {
    let model = DeepLearningModel()
    if let prediction = model.predict(input: input) {
        print("Prediction: \(prediction)")
    }
}
