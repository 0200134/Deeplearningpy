use tch::{nn, Device, Tensor, no_grad};
use tch::nn::{Module, OptimizerConfig};

fn main() {
    // Set device to CPU or CUDA if available
    let device = Device::cuda_if_available();
    println!("Running on: {:?}", device);

    // Create a neural network
    let vs = nn::VarStore::new(device);
    let net = nn::seq()
        .add(nn::linear(&vs.root(), 28 * 28, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root(), 128, 10, Default::default()));

    // Generate random data
    let input = Tensor::randn(&[64, 28 * 28], (tch::Kind::Float, device));
    let target = Tensor::randn(&[64, 10], (tch::Kind::Float, device));

    // Forward pass
    let output = net.forward(&input);

    // Calculate loss
    let loss = output.mse_loss(&target, tch::Reduction::Mean);
    println!("Loss: {}", f64::from(loss));

    // Backward pass and update weights
    let mut optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
    optimizer.backward_step(&loss);
}
