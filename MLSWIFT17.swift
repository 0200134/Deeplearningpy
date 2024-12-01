import CoreML
import Vision

class ImageClassifier {
    let model = try! VNCoreMLModel(for: YourModel.init())

    func classifyImage(image: CIImage) {
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNClassificationObservation] else {
                return
            }
            let topClassification = results.first!
            print(topClassification.identifier, topClassification.confidence)
        }
        let handler = VNImageRequestHandler(ciImage: image)
        try? handler.perform([request])
    }
}
