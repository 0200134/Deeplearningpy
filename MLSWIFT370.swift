import Foundation

// Define a function to perform a basic linear regression
func linearRegression(input: [Double], target: [Double]) -> (Double, Double) {
    let n = Double(input.count)
    let sumX = input.reduce(0, +)
    let sumY = target.reduce(0, +)
    let sumXY = zip(input, target).reduce(0) { $0 + $1.0 * $1.1 }
    let sumX2 = input.reduce(0) { $0 + $1 * $1 }
    
    let slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
    let intercept = (sumY - slope * sumX) / n
    
    return (slope, intercept)
}

// Example usage
let input = [1.0, 2.0, 3.0, 4.0, 5.0]
let target = [2.0, 4.0, 6.0, 8.0, 10.0]

let (slope, intercept) = linearRegression(input: input, target: target)
print("Slope: \(slope), Intercept: \(intercept)")
