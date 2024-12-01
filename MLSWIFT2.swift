import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var classificationLabel: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()
    }

    @IBAction func pickImage(_ sender: Any) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true, completion: nil)
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true, completion: nil)
        guard let image = info[.originalImage] as? UIImage else { return }
        imageView.image = image
        classifyImage(image)
    }

    func classifyImage(_ image: UIImage) {
        guard let ciImage = CIImage(image: image) else { return }
        guard let model = try? VNCoreMLModel(for: MobileNetV2().model) else { return }

        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let firstResult = results.first else {
                self?.classificationLabel.text = "Could not classify image."
                return
            }
            DispatchQueue.main.async {
                self?.classificationLabel.text = "\(firstResult.identifier) - \(firstResult.confidence * 100)%"
            }
        }

        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async {
                    self.classificationLabel.text = "Error: \(error.localizedDescription)"
                }
            }
        }
    }
}func classifyImage(_ image: UIImage) {
    guard let ciImage = CIImage(image: image) else {
        self.classificationLabel.text = "Unable to create CIImage."
        return
    }
    guard let model = try? VNCoreMLModel(for: MobileNetV2().model) else {
        self.classificationLabel.text = "Failed to load model."
        return
    }

    let request = VNCoreMLRequest(model: model) { [weak self] request, error in
        guard let results = request.results as? [VNClassificationObservation],
              let firstResult = results.first else {
            self?.classificationLabel.text = "Could not classify image."
            return
        }
        DispatchQueue.main.async {
            self?.classificationLabel.text = "\(firstResult.identifier) - \(firstResult.confidence * 100)%"
        }
    }

    let handler = VNImageRequestHandler(ciImage: ciImage, orientation: .up, options: [:])
    DispatchQueue.global(qos: .userInitiated).async {
        do {
            try handler.perform([request])
        } catch {
            DispatchQueue.main.async {
                self.classificationLabel.text = "Error: \(error.localizedDescription)"
            }
        }
    }
}
