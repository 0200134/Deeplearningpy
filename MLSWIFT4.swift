import UIKit
import CoreML
import Vision
import AVFoundation

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var classificationLabel: UILabel!
    @IBOutlet weak var modelSelector: UISegmentedControl!

    var captureSession: AVCaptureSession!

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupModelSelector()
    }

    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo

        guard let captureDevice = AVCaptureDevice.default(for: .video) else { return }
        guard let input = try? AVCaptureDeviceInput(device: captureDevice) else { return }
        captureSession.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(output)

        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.frame
        view.layer.insertSublayer(previewLayer, at: 0)

        captureSession.startRunning()
    }

    func setupModelSelector() {
        modelSelector.addTarget(self, action: #selector(modelChanged), for: .valueChanged)
    }

    @objc func modelChanged() {
        // Update UI or model state based on selected segment
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let selectedModel: VNCoreMLModel
        switch modelSelector.selectedSegmentIndex {
        case 0:
            selectedModel = try! VNCoreMLModel(for: MobileNetV2().model)
        case 1:
            selectedModel = try! VNCoreMLModel(for: ResNet50().model)
        default:
            return
        }

        let request = VNCoreMLRequest(model: selectedModel) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let firstResult = results.first else { return }

            DispatchQueue.main.async {
                self?.classificationLabel.text = "\(firstResult.identifier) - \(firstResult.confidence * 100)%"
            }
        }

       ImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
}func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        self.classificationLabel.text = "Could not get pixel buffer."
        return
    }

    let selectedModel: VNCoreMLModel
    switch modelSelector.selectedSegmentIndex {
    case 0:
        selectedModel = try! VNCoreMLModel(for: MobileNetV2().model)
    case 1:
        selectedModel = try! VNCoreMLModel(for: ResNet50().model)
    default:
        self.classificationLabel.text = "No model selected."
        return
    }

    let request = VNCoreMLRequest(model: selectedModel) { [weak self] request, error in
        guard let results = request.results as? [VNClassificationObservation],
              let firstResult = results.first else {
            DispatchQueue.main.async {
                self?.classificationLabel.text = "Could not classify image."
            }
            return
        }

        DispatchQueue.main.async {
            self?.classificationLabel.text = "\(firstResult.identifier) - \(firstResult.confidence * 100)%"
        }
    }

    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
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
