extern crate tch;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor, vision};

fn main() -> failure::Fallible<()> {
    // Set the device to use for training (CUDA or CPU)
    let device = Device::cuda_if_available();

    // Load the MNIST dataset
    let mut train_data = vision::mnist::load_dir("data/mnist")?;
    let test_data = vision::mnist::load_dir("data/mnist")?;

    // Build the model
    let vs = nn::VarStore::new(device);
    let model = build_model(&vs.root());

    // Train the model
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 1..=10 {
        train_model(&model, &mut train_data, &mut opt, epoch)?;
        test_model(&model, &test_data)?;
    }

    Ok(())
}fn build_model(vs: &nn::Path) -> impl nn::Module {
    nn::seq()
        .add(nn::conv2d(vs, 1, 32, 5, Default::default()))
        .add_fn(|xs| xs.max_pool2d_default(2))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 32, 64, 5, Default::default()))
        .add_fn(|xs| xs.max_pool2d_default(2))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.view([-1, 1024]))
        .add(nn::linear(vs, 1024, 1024, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 1024, 10, Default::default()))
}fn train_model(
    model: &impl nn::Module,
    train_data: &mut vision::mnist::Dataset,
    opt: &mut nn::Optimizer<nn::Adam>,
    epoch: i64,
) -> failure::Fallible<()> {
    for (b, (images, labels)) in train_data.train_iter(64).enumerate() {
        let logits = model.forward(&images);
        let loss = logits.cross_entropy_for_logits(&labels);
        opt.backward_step(&loss);

        if b % 100 == 0 {
            println!("epoch: {:4} batch: {:4} loss: {:8.6}", epoch, b, f64::from(&loss));
        }
    }
    Ok(())
}

fn test_model(
    model: &impl nn::Module,
    test_data: &vision::mnist::Dataset,
) -> failure::Fallible<()> {
    let mut correct = 0;
    let mut total = 0;

    for (images, labels) in test_data.test_iter(1000) {
        let logits = model.forward(&images);
        let predicted = logits.argmax1(-1, false);
        correct += predicted.eq1(&labels).sum().int64_value(&[]);
        total += labels.size()[0];
    }

    let accuracy = 100.0 * correct as f64 / total as f64;
    println!("Accuracy: {:5.2}%", accuracy);

    Ok(())
}
