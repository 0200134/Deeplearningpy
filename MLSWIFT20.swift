import Foundation
import CoreML
import Vision
import UIKit

class ImageClassifier {
    let model: VNCoreMLModel

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

    func classify(image: UIImage, completion: @escaping (String) -> Void) {
        guard let ciImage = CIImage(image: image) else {
            fatalError("Unable to create CIImage from UIImage")
        }

        let request = VNCoreMLRequest(model: self.model) { (request, error) in
            guard let results = request.results as? [VNClassificationObservation], let topResult = results.first else {
                fatalError("Unexpected results")
            }
            completion(topResult.identifier)
        }

        let handler = VNImageRequestHandler(ciImage: ciImage)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([request])
            } catch {
                print("Failed to perform classification: \(error)")
            }
        }
    }
}

// Example usage
let imageClassifier = ImageClassifier(modelName: "YourModelName")

if let image = UIImage(named: "example.jpg") {
    imageClassifier.classify(image: image) { result in
        print("Classification: \(result)")
    }
}func classify(image: UIImage, completion: @escaping ([VNClassificationObservation]) -> Void) {
    guard let ciImage = CIImage(image: image) else {
        fatalError("Unable to create CIImage from UIImage")
    }

    let request = VNCoreMLRequest(model: self.model) { (request, error) in
        guard let results = request.results as? [VNClassificationObservation] else {
            fatalError("Unexpected results")
        }
        completion(results)
    }

    let handler = VNImageRequestHandler(ciImage: ciImage)
    DispatchQueue.global(qos: .userInteractive).async {
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform classification: \(error)")
        }
    }
}

// Example usage for multiple results
imageClassifier.classify(image: image) { results in
    results.forEach { result in
        print("Classification: \(result.identifier) - Confidence: \(result.confidence)")
    }
}func classify(image: UIImage, completion: @escaping (Result<[VNClassificationObservation], Error>) -> Void) {
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

// Example usage with error handling
imageClassifier.classify(image: image) { result in
    switch result {
    case .success(let results):
        results.forEach { result in
            print("Classification: \(result.identifier) - Confidence: \(result.confidence)")
        }
    case .failure(let error):
        print("Classification failed: \(error)")
    }
}
