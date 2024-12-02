import CoreML
import UIKit

class ViewController: UIViewController {
    var model: VNCoreMLModel?

    override func viewDidLoad() {
        super.viewDidLoad()
        
        guard let model = try? VNCoreMLModel(for: YourModel().model) else {
            fatalError("Can't load the ML model")
        }
        
        self.model = model
    }

    func predict(image: UIImage) {
        guard let pixelBuffer = image.toCVPixelBuffer() else {
            fatalError("Image to CVPixelBuffer conversion failed")
        }

        let request = VNCoreMLRequest(model: model!) { (request, error) in
            guard let results = request.results as? [VNClassificationObservation] else {
                fatalError("Unexpected result type from VNCoreMLRequest")
            }
            for classification in results {
                print("Classification: \(classification.identifier) Confidence: \(classification.confidence)")
            }
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
}

// Helper extension to convert UIImage to CVPixelBuffer
extension UIImage {
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let image = self.cgImage!
        let width = image.width
        let height = image.height
        let attributes = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                          kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attributes, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        let data = CVPixelBufferGetBaseAddress(buffer)!
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: data, width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        return buffer
    }
}
