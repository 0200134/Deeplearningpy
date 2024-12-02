import Foundation
import UIKit

// Define an enum for API errors
enum APIError: Error {
    case responseError
    case dataError
}

// Function to generate images from text prompt using OpenAI's API
func generateImages(prompt: String, n: Int = 1, size: String = "512x512", completion: @escaping (Result<[String], APIError>) -> Void) {
    let apiKey = "YOUR_API_KEY_HERE"
    let urlString = "https://api.openai.com/v1/images/generations"
    
    guard let url = URL(string: urlString) else { return }
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.addValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    request.addValue("application/json", forHTTPHeaderField: "Content-Type")
    
    let body: [String: Any] = [
        "prompt": prompt,
        "n": n,
        "size": size
    ]
    request.httpBody = try? JSONSerialization.data(withJSONObject: body)
    
    URLSession.shared.dataTask(with: request) { data, response, error in
        if let error = error {
            print("Error making request: \(error.localizedDescription)")
            completion(.failure(.responseError))
            return
        }
        
        guard let data = data else {
            completion(.failure(.dataError))
            return
        }
        
        do {
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let images = json["data"] as? [[String: Any]] {
                let imageUrls = images.compactMap { $0["url"] as? String }
                completion(.success(imageUrls))
            } else {
                completion(.failure(.dataError))
            }
        } catch {
            completion(.failure(.dataError))
        }
    }.resume()
}

// Function to save images locally
func saveImages(imageUrls: [String], directory: String = "generated_images") {
    let fileManager = FileManager.default
    let documentsUrl = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
    let directoryUrl = documentsUrl.appendingPathComponent(directory)
    
    do {
        if !fileManager.fileExists(atPath: directoryUrl.path) {
            try fileManager.createDirectory(at: directoryUrl, withIntermediateDirectories: true, attributes: nil)
        }
        
        for (index, imageUrl) in imageUrls.enumerated() {
            guard let url = URL(string: imageUrl), let data = try? Data(contentsOf: url), let image = UIImage(data: data) else {
                print("Error downloading image from \(imageUrl)")
                continue
            }
            let imagePath = directoryUrl.appendingPathComponent("image_\(index + 1).png")
            try image.pngData()?.write(to: imagePath)
            print("Image saved at \(imagePath.path)")
        }
    } catch {
        print("Error saving images: \(error.localizedDescription)")
    }
}

// Example usage
let prompt = "A futuristic cityscape with flying cars and neon lights"
generateImages(prompt: prompt, n: 3) { result in
    switch result {
    case .success(let imageUrls):
        saveImages(imageUrls: imageUrls)
    case .failure(let error):
        print("Failed to generate images: \(error)")
    }
}
