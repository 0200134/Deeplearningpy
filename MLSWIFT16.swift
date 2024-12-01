import CoreML
import Vision
import UIKit

class ImageClassifier {
    let model: VNCoreMLModel

    init() {
        guard let coreMLModel = try? MobileNetV2(configuration: MLModelConfiguration()).model else {
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
}let imageClassifier = ImageClassifier()

if let image = UIImage(named: "example.jpg") {
    imageClassifier.classify(image: image) { result in
        print("Classification: \(result)")
    }
}
