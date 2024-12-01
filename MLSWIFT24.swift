import Foundation
import CoreML
import Vision
import UIKit

class ImageClassifier {
    private let model: VNCoreMLModel

    // Initialize with the CoreML model
    init(modelName: String) {
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            fatalError("Model not found")
        }
        guard let coreMLModel = try? MLModel(contentsOf: modelURL) else {
            fatalError("Failed to load CoreML model")
        }
        guard let vnModel = try? VNCoreMLModel(for: coreMLModel) else {
            fatalError("Failed to create VNCoreMLModel")
        }
        self.model = vnModel
    }

    // Classify the given image
    func classify(image: UIImage, completion: @escaping (Result<[VNClassificationObservation], Error>) -> Void) {
        guard let ciImage = CIImage(image: image) else {
            fatalError("Unable to create CIImage from UIImage")
        }

        let request = VNCoreMLRequest(model: self.model) { (request, error) in
            if let error = error {
                completion(.failure(error))
                return
            }

            guard let results = request.results as? [VNClassificationObservation] else {
                fatalError("Unexpected results")
            }
            completion(.success(results))
        }

        let handler = VNImageRequestHandler(ciImage: ciImage)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([request])
            } catch {
                completion(.failure(error))
            }
        }
    }
}

// Example
