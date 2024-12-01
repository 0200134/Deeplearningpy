import Foundation
import CoreML

class AIModel {
    let model: MLModel

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

let model = AIModel(modelName: "YourModelName")
