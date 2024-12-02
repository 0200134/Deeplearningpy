import CoreML
import UIKit

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    // Instance of the model
    var model: MobileNetV2?

    override func viewDidLoad() {
        super.viewDidLoad()
        // Load the model
        do {
            model = try MobileNetV2(configuration: MLModelConfiguration())
        } catch {
            print("Error loading model: \(error)")
        }
    }

    // Function to make predictions on the image
    func classifyImage(image: UIImage) {
        guard let pixelBuffer = image.toCVPixelBuffer() else {
            fatalError("Image conversion to pixel buffer failed")
        }

        guard let prediction = try? model?.prediction(image: pixelBuffer) else {
            fatalError("Prediction failed")
        }

        let label = prediction.classLabel
        print("Classified as: \(label)")
    }
}
extension UIImage {
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let width = Int(self.size.width)
        let height = Int(self.size.height)
        let attributes: [NSObject: AnyObject] = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ]

        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attributes as CFDictionary, &pixelBuffer)

        guard let buffer = pixelBuffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        let pixelData = CVPixelBufferGetBaseAddress(buffer)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        context?.translateBy(x: 0, y: CGFloat(height))
        context?.scaleBy(x: 1.0, y: -1.0)

        UIGraphicsPushContext(context!)
        self.draw(in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)

        return buffer
    }
}
extension ViewController {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
        picker.dismiss(animated: true, completion: nil)
        guard let image = info[.originalImage] as? UIImage else {
            return
        }
        classifyImage(image: image)
    }
}
@IBAction func pickImage(_ sender: Any) {
    let imagePicker = UIImagePickerController()
    imagePicker.delegate = self
    imagePicker.sourceType = .photoLibrary
    present(imagePicker, animated: true, completion: nil)
}

@IBOutlet weak var classificationLabel: UILabel!

func classifyImage(image: UIImage) {
    // Existing classification code
    let label = prediction.classLabel
    classificationLabel.text = "Classified as: \(label)"
}
