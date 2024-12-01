import UIKit
import CoreML
import Vision

class ViewController: UIViewController {

    // Reference to the image view
    @IBOutlet weak var imageView: UIImageView!

    // Reference to the label
    @IBOutlet weak var classificationLabel: UILabel!

    // The model
    let model = YourModelName() // Replace with your model's class name

    override func viewDidLoad() {
        super.viewDidLoad()

        // Classify an example image
        if let image = UIImage(named: "example.jpg") {
            classifyImage(image)
        }
    }

    func classifyImage(_ image: UIImage) {
        // Convert the image to a CIImage
        guard let ciImage = CIImage(image: image) else {
            fatalError("Unable to create \(CIImage.self) from \(image).")
        }

        // Perform the classification
        let handler = VNImageRequestHandler(ciImage: ciImage)
        do {
            let request = VNCoreMLRequest(model: VNCoreMLModel(for: model.model)) { request, error in
                if let results = request.results as? [VNClassificationObservation] {
                    self.updateClassificationLabel(results)
                }
            }
            try handler.perform([request])
        } catch {
            print("Failed to perform classification.\n\(error.localizedDescription)")
        }
    }

    func updateClassificationLabel(_ classifications: [VNClassificationObservation]) {
        if let topResult = classifications.first {
            DispatchQueue.main.async {
                self.classificationLabel.text = topResult.identifier + " (\(topResult.confidence))"
            }
        }
    }
}
