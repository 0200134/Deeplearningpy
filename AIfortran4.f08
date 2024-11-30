module neural_network
  implicit none
  integer, parameter :: input_size = 3072 ! CIFAR-10 images are 32x32x3 = 3072
  integer, parameter :: hidden_size = 128
  integer, parameter :: output_size = 10
  real, dimension(input_size, hidden_size) :: weights1
  real, dimension(hidden_size, output_size) :: weights2
  real, dimension(hidden_size) :: bias1
  real, dimension(output_size) :: bias2
  real, parameter :: learning_rate = 0.001

contains

  subroutine initialize()
    integer :: i, j
    call random_number(weights1)
    call random_number(weights2)
    call random_number(bias1)
    call random_number(bias2)
    weights1 = weights1 * 0.01
    weights2 = weights2 * 0.01
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

  function forward(input, hidden_output) result(output)
    real, dimension(input_size), intent(in) :: input
    real, dimension(hidden_size), intent(out) :: hidden_output
    real, dimension(output_size) :: output
    hidden_output = matmul(input, weights1) + bias1
    hidden_output = max(0.0, hidden_output) ! ReLU activation
    output = matmul(hidden_output, weights2) + bias2
    output = softmax(output)
  end function forward

  subroutine backward(input, hidden_output, output, labels)
    real, dimension(input_size), intent(in) :: input
    real, dimension(hidden_size), intent(in) :: hidden_output
    real, dimension(output_size), intent(in) :: output
    real, dimension(output_size), intent(in) :: labels
    real, dimension(output_size) :: output_error
    real, dimension(hidden_size) :: hidden_error

    output_error = output - labels

    ! Update weights and biases for the second layer
    weights2 = weights2 - learning_rate * matmul(reshape(hidden_output, [hidden_size, 1]), reshape(output_error, [1, output_size]))
    bias2 = bias2 - learning_rate * output_error

    ! Calculate hidden error
    hidden_error = matmul(output_error, transpose(weights2))
    hidden_error = hidden_error * merge(1.0, 0.0, hidden_output > 0.0) ! ReLU derivative

    ! Update weights and biases for the first layer
    weights1 = weights1 - learning_rate * matmul(reshape(input, [input_size, 1]), reshape(hidden_error, [1, hidden_size]))
    bias1 = bias1 - learning_rate * hidden_error
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
  real, dimension(hidden_size) :: hidden_output
  integer :: i, epochs

  call initialize()

  ! Load your data into the input and labels arrays here
  ! Example: input = ...
  ! Example: labels = ...

  epochs = 1000
  do i = 1, epochs
    output = forward(input, hidden_output)
    call backward(input, hidden_output, output, labels)
    if (mod(i, 100) == 0) then
      print *, "Epoch: ", i, " Loss: ", compute_loss(output, labels)
    end if
  end do

  print *, "Training complete."
end program main
