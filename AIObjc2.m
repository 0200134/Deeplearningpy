#import "MNISTClassifierBridge.h"

@implementation MNISTClassifierBridge

+ (NSString *)predictDigit:(UIImage *)image {
    MNISTClassifier *model = [[MNISTClassifier alloc] init];
    NSError *error = nil;

    CVPixelBufferRef buffer = [self pixelBufferFromImage:image];
    if (!buffer) {
        return @"Failed to create pixel buffer.";
    }

    MNISTClassifierOutput *output = [model predictionFromImage:buffer error:&error];
    if (error) {
        return [NSString stringWithFormat:@"Prediction failed: %@", error.localizedDescription];
    }

    return [NSString stringWithFormat:@"Predicted digit: %@", output.classLabel];
}

+ (CVPixelBufferRef)pixelBufferFromImage:(UIImage *)image {
    CGImageRef cgImage = [image CGImage];
    NSDictionary *options = @{ (id)kCVPixelBufferCGImageCompatibilityKey: @YES,
                               (id)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES };

    CVPixelBufferRef pxbuffer = NULL;
    CGFloat frameWidth = CGImageGetWidth(cgImage);
    CGFloat frameHeight = CGImageGetHeight(cgImage);

    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, frameWidth, frameHeight,
                                          kCVPixelFormatType_32ARGB, (__bridge CFDictionaryRef)options,
                                          &pxbuffer);

    NSParameterAssert(status == kCVReturnSuccess && pxbuffer != NULL);

    CVPixelBufferLockBaseAddress(pxbuffer, 0);
    void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);
    NSParameterAssert(pxdata != NULL);

    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(pxdata, frameWidth, frameHeight,
                                                 8, 4*frameWidth, rgbColorSpace,
                                                 kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
    NSParameterAssert(context);

    CGContextDrawImage(context, CGRectMake(0, 0, frameWidth, frameHeight), cgImage);
    CGColorSpaceRelease(rgbColorSpace);
    CGContextRelease(context);

    CVPixelBufferUnlockBaseAddress(pxbuffer, 0);

    return pxbuffer;
}
@end
