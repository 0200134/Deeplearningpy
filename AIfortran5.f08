module neural_network
  implicit none
  integer, parameter :: input_size = 3072 ! CIFAR-10 images are 32x32x3 = 3072
  integer, parameter :: hidden_size1 = 256
  integer, parameter :: hidden_size2 = 128
  integer, parameter :: output_size = 10
  integer, parameter :: batch_size = 64
  integer, parameter :: num_batches = 100
  real, parameter :: learning_rate = 0.001
  real, parameter :: momentum = 0.9
  real, dimension(input_size, hidden_size1) :: weights1
  real, dimension(hidden_size1, hidden_size2) :: weights2
  real, dimension(hidden_size2, output_size) :: weights3
  real, dimension(hidden_size1) :: bias1
  real, dimension(hidden_size2) :: bias2
  real, dimension(output_size) :: bias3
  real, dimension(input_size, hidden_size1) :: velocity_w1
  real, dimension(hidden_size1, hidden_size2) :: velocity_w2
  real, dimension(hidden_size2, output_size) :: velocity_w3
  real, dimension(hidden_size1) :: velocity_b1
  real, dimension(hidden_size2) :: velocity_b2
  real, dimension(output_size) :: velocity_b3

contains

  subroutine initialize()
    integer :: i, j
    call random_number(weights1)
    call random_number(weights2)
    call random_number(weights3)
    call random_number(bias1)
    call random_number(bias2)
    call random_number(bias3)
    weights1 = weights1 * 0.01
    weights2 = weights2 * 0.01
    weights3 = weights3 * 0.01
    velocity_w1 = 0.0
    velocity_w2 = 0.0
    velocity_w3 = 0.0
    velocity_b1 = 0.0
    velocity_b2 = 0.0
    velocity_b3 = 0.0
  end subroutine initialize

  function relu(x) result(y)
    real, intent(in) :: x
    real :: y
    y = max(0.0, x)
  end function relu

  function relu_derivative(x) result(y)
    real, intent(in) :: x
    real :: y
    y = merge(1.0, 0.0, x > 0.0)
  end function relu_derivative

  function softmax(x) result(y)
    real, dimension(:), intent(in) :: x
    real, dimension(size(x)) :: y
    real :: sum_exp
    integer :: i
    sum_exp = sum(exp(x))
    y = exp(x) / sum_exp
  end function softmax

  function forward(input, hidden_output1, hidden_output2) result(output)
    real, dimension(input_size), intent(in) :: input
    real, dimension(hidden_size1), intent(out) :: hidden_output1
    real, dimension(hidden_size2), intent(out) :: hidden_output2
    real, dimension(output_size) :: output
    hidden_output1 = matmul(input, weights1) + bias1
    hidden_output1 = max(0.0, hidden_output1) ! ReLU activation
    hidden_output2 = matmul(hidden_output1, weights2) + bias2
    hidden_output2 = max(0.0, hidden_output2) ! ReLU activation
    output = matmul(hidden_output2, weights3) + bias3
    output = softmax(output)
  end function forward

  subroutine backward(input, hidden_output1, hidden_output2, output, labels)
    real, dimension(input_size), intent(in) :: input
    real, dimension(hidden_size1), intent(in) :: hidden_output1
    real, dimension(hidden_size2), intent(in) :: hidden_output2
    real, dimension(output_size), intent(in) :: output
    real, dimension(output_size), intent(in) :: labels
    real, dimension(output_size) :: output_error
    real, dimension(hidden_size2) :: hidden_error2
    real, dimension(hidden_size1) :: hidden_error1

    output_error = output - labels

    ! Update weights and biases for the third layer
    velocity_w3 = momentum * velocity_w3 + learning_rate * matmul(reshape(hidden_output2, [hidden_size2, 1]), reshape(output_error, [1, output_size]))
    weights3 = weights3 - velocity_w3
    velocity_b3 = momentum * velocity_b3 + learning_rate * output_error
    bias3 = bias3 - velocity_b3

    ! Calculate hidden error for the second layer
    hidden_error2 = matmul(output_error, transpose(weights3))
    hidden_error2 = hidden_error2 * merge(1.0, 0.0, hidden_output2 > 0.0) ! ReLU derivative

    ! Update weights and biases for the second layer
    velocity_w2 = momentum * velocity_w2 + learning_rate * matmul(reshape(hidden_output1, [hidden_size1, 1]), reshape(hidden_error2, [1, hidden_size2]))
    weights2 = weights2 - velocity_w2
    velocity_b2 = momentum * velocity_b2 + learning_rate * hidden_error2
    bias2 = bias2 - velocity_b2

    ! Calculate hidden error for the first layer
    hidden_error1 = matmul(hidden_error2, transpose(weights2))
    hidden_error1 = hidden_error1 * merge(1.0, 0.0, hidden_output1 > 0.0) ! ReLU derivative

    ! Update weights and biases for the first layer
    velocity_w1 = momentum * velocity_w1 + learning_rate * matmul(reshape(input, [input_size, 1]), reshape(hidden_error1, [1, hidden_size1]))
    weights1 = weights1 - velocity_w1
    velocity_b1 = momentum * velocity_b1 + learning_rate * hidden_error1
    bias1 = bias1 - velocity_b1
  end subroutine backward

  function compute_loss(output, labels) result(loss)
    real, dimension(:), intent(in) :: output, labels
    real :: loss
    loss = -sum(labels * log(output))
  end function compute_loss

end module neural_network

program main
  use neural_network
  implicit none
  real, dimension(input_size) :: input
  real, dimension(output_size) :: labels, output
  real, dimension(hidden_size1) :: hidden_output1
  real, dimension(hidden_size2) :: hidden_output2
  integer :: i, j, epochs

  call initialize()

  ! Load your data into the input and labels arrays here
  ! Example: input = ...
  ! Example: labels = ...

  epochs = 1000
  do i = 1, epochs
    do j = 1, num_batches
      ! Load batch data into input and labels here
      ! Example: input = ...
      ! Example: labels = ...
      output = forward(input, hidden_output1, hidden_output2)
      call backward(input, hidden_output1, hidden_output2, output, labels)
    end do
    if (mod(i, 100) == 0) then
      print *, "Epoch: ", i, " Loss: ", compute_loss(output, labels)
    end if
  end do

  print *, "Training complete."
end program main
