import NaturalLanguage

func classifyText(_ text: String) {
    // Load a pre-trained NLModel for text classification
    guard let modelURL = Bundle.main.url(forResource: "TextClassifier", withExtension: "mlmodelc"),
          let model = try? NLModel(contentsOf: modelURL) else {
        fatalError("Failed to load model")
    }

    // Perform classification
    let predictedLabel = model.predictedLabel(for: text)
    print("Predicted label: \(String(describing: predictedLabel))")
}

let sampleText = "This is an example text for classification."
classifyText(sampleText)
