#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

// Define the MLP structure
@interface MLP : NSObject
@property (nonatomic, strong) simd::float4x4 fc1;
@property (nonatomic, strong) simd::float4x4 fc2;
@property (nonatomic, strong) simd::float4x4 fc3;

- (instancetype)init;
- (simd::float4)forward:(simd::float4)input;
- (void)train:(NSArray<simd::float4> *)inputs targets:(NSArray<simd::float4> *)targets epochs:(int)epochs learningRate:(float)learningRate;
@end

@implementation MLP

- (instancetype)init {
    self = [super init];
    if (self) {
        // Initialize weights with random values
        self.fc1 = matrix_identity_float4x4;
        self.fc2 = matrix_identity_float4x4;
        self.fc3 = matrix_identity_float4x4;
    }
    return self;
}

- (simd::float4)forward:(simd::float4)input {
    simd::float4 x = simd_relu(simd_mul(input, self.fc1));
    x = simd_relu(simd_mul(x, self.fc2));
    x = simd_relu(simd_mul(x, self.fc3));
    return x;
}

- (void)train:(NSArray<simd::float4> *)inputs targets:(NSArray<simd::float4> *)targets epochs:(int)epochs learningRate:(float)learningRate {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < inputs.count; i++) {
            simd::float4 input = inputs[i];
            simd::float4 target = targets[i];

            // Forward pass
            simd::float4 output = [self forward:input];

            // Compute loss (mean squared error)
            simd::float4 loss = simd_sub(output, target);
            loss = simd_mul(loss, loss);
            float lossValue = simd_dot(loss, simd::float4(1.0f));

            // Backpropagation (simplified gradient descent)
            simd::float4 gradient = simd_mul(simd_sub(output, target), learningRate);
            self.fc3 = simd_add(self.fc3, simd_outer_product(simd_relu(simd_mul(input, self.fc2)), gradient));
            self.fc2 = simd_add(self.fc2, simd_outer_product(simd_relu(input), gradient));
            self.fc1 = simd_add(self.fc1, simd_outer_product(input, gradient));
        }
        NSLog(@"Completed epoch %d", epoch);
    }
}

@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // Initialize MLP and data
        MLP *mlp = [[MLP alloc] init];
        NSArray<simd::float4> *inputs = @[
            simd::float4(0.5, 1.5, 1.0, 2.0),
            simd::float4(1.5, 3.0, 0.5, 1.0)
        ];
        NSArray<simd::float4> *targets = @[
            simd::float4(1.0, 0.0, 0.0, 0.0),
            simd::float4(0.0, 1.0, 0.0, 0.0)
        ];
        
        // Train the model
        [mlp train:inputs targets:targets epochs:10 learningRate:0.01];
        
        // Forward pass with new input
        simd::float4 newInput = simd::float4(0.7, 1.2, 1.0, 1.5);
        simd::float4 prediction = [mlp forward:newInput];
        
        NSLog(@"Prediction: %f %f %f %f", prediction.x, prediction
