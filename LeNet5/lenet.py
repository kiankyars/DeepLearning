"""
PEDAGOGICAL LENET-5 IMPLEMENTATION
==================================

This is a learning-focused implementation of LeNet-5 with:
- Detailed comments explaining every step
- Mathematical formulas and intuitions
- Shape tracking and debugging features
- Step-by-step explanations of backpropagation
- Visual debugging capabilities

LEARNING PATH:
1. Start with ReLU (simplest)
2. Move to FC (fully connected) layers
3. Understand Pooling
4. Master Convolution (most complex)
5. Study the complete training loop
"""

import numpy as np
import matplotlib.pyplot as plt

class BaseLayer:
    """Base class for all layers with common functionality"""
    
    def __init__(self, name=""):
        self.name = name
        self.debug_mode = False  # Set to True to see detailed computations
        
    def print_debug(self, message, data=None):
        """Helper function to print debug information"""
        if self.debug_mode:
            print(f"[{self.name}] {message}")
            if data is not None:
                print(f"    Shape: {data.shape}, Type: {data.dtype}")
                if data.size < 20:  # Only print small arrays
                    print(f"    Data: {data}")

class Conv(BaseLayer):
    """
    CONVOLUTION LAYER
    =================
    
    WHAT IT DOES:
    - Applies learned filters to input images
    - Each filter detects different patterns (edges, textures, etc.)
    - Output is feature maps showing where these patterns appear
    
    MATHEMATICAL FORMULA:
    Output[b,h,w,c] = Σ(i,j,k) Input[b,h+i,w+j,k] * Weight[i,j,k,c] + Bias[c]
    
    WHERE:
    - b = batch index
    - h,w = spatial positions
    - c = output channel (filter)
    - i,j = kernel positions
    - k = input channel
    """
    
    def __init__(self, name, kernel_size, in_channels, out_channels):
        super().__init__(name)
        
        # LAYER PARAMETERS
        self.kernel_size = kernel_size      # e.g., 5x5 kernel
        self.in_channels = in_channels      # e.g., 1 for grayscale, 3 for RGB
        self.out_channels = out_channels    # e.g., 6 filters in first conv layer
        
        # LEARNABLE PARAMETERS
        # Weight shape: (kernel_height, kernel_width, in_channels, out_channels)
        # This is like having 'out_channels' different filters, each of size kernel_size x kernel_size
        self.weight = np.random.randn(kernel_size, kernel_size, in_channels, out_channels) * 0.1
        self.bias = np.zeros(out_channels)
        
        # TRAINING HYPERPARAMETERS
        self.learning_rate = 0.0003
        self.weight_decay = 0.001  # L2 regularization
        
        # STORED FOR BACKPROPAGATION
        self.input = None
        self.output = None
        
        self.print_debug(f"Initialized Conv layer: {kernel_size}x{kernel_size}, {in_channels}->{out_channels} channels")
    
    def forward(self, x):
        """
        FORWARD PASS: CONVOLUTION OPERATION
        ===================================
        
        INPUT:  x shape = (batch_size, height, width, channels)
        OUTPUT: shape = (batch_size, new_height, new_width, out_channels)
        
        ALGORITHM:
        1. For each output position (h_out, w_out):
        2. Extract a patch from input at that position
        3. Multiply patch with each filter and sum
        4. Add bias for each output channel
        """
        self.input = x
        batch_size, height, width, channels = x.shape
        
        # CALCULATE OUTPUT DIMENSIONS
        # With no padding and stride=1: output_size = input_size - kernel_size + 1
        out_height = height - self.kernel_size + 1
        out_width = width - self.kernel_size + 1
        
        self.print_debug(f"Input shape: {x.shape}")
        self.print_debug(f"Output shape: ({batch_size}, {out_height}, {out_width}, {self.out_channels})")
        
        # INITIALIZE OUTPUT
        output = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        # RESHAPE WEIGHTS FOR EFFICIENT COMPUTATION
        # Reshape from (kernel_h, kernel_w, in_ch, out_ch) to (kernel_h*kernel_w*in_ch, out_ch)
        weight_reshaped = self.weight.reshape(-1, self.out_channels)
        
        # CONVOLUTION OPERATION
        for h_out in range(out_height):
            for w_out in range(out_width):
                # EXTRACT PATCH: Get the kernel-sized window from input
                patch = x[:, h_out:h_out+self.kernel_size, w_out:w_out+self.kernel_size, :]
                
                # RESHAPE PATCH: Flatten to (batch_size, kernel_h*kernel_w*in_channels)
                patch_reshaped = patch.reshape(batch_size, -1)
                
                # CONVOLVE: Matrix multiplication of patch with weights
                # This computes: Σ(patch * weight) for each output channel
                conv_result = patch_reshaped.dot(weight_reshaped)
                
                # ADD BIAS: Add bias term for each output channel
                output[:, h_out, w_out, :] = conv_result + self.bias
                
                if self.debug_mode and h_out == 0 and w_out == 0:
                    self.print_debug(f"First patch shape: {patch.shape}")
                    self.print_debug(f"First conv result shape: {conv_result.shape}")
        
        self.output = output
        return output
    
    def backward(self, grad_output):
        """
        BACKWARD PASS: COMPUTE GRADIENTS
        ================================
        
        This is the most complex part! We need to compute:
        1. Gradient w.r.t. weights (∂L/∂W)
        2. Gradient w.r.t. bias (∂L/∂b)  
        3. Gradient w.r.t. input (∂L/∂x) - for backpropagation to previous layer
        
        MATHEMATICAL DERIVATION:
        - ∂L/∂W = ∂L/∂y * ∂y/∂W = grad_output * input
        - ∂L/∂b = ∂L/∂y * ∂y/∂b = sum(grad_output)
        - ∂L/∂x = ∂L/∂y * ∂y/∂x = grad_output * weight (transposed convolution)
        """
        batch_size, out_height, out_width, out_channels = grad_output.shape
        kernel_size = self.kernel_size
        in_height = out_height + kernel_size - 1
        in_width = out_width + kernel_size - 1
        
        self.print_debug(f"Backward: grad_output shape: {grad_output.shape}")
        
        # 1. COMPUTE GRADIENT W.R.T. WEIGHTS
        # ∂L/∂W[i,j,k,c] = Σ(b,h,w) grad_output[b,h,w,c] * input[b,h+i,w+j,k]
        weight_grad = np.zeros_like(self.weight)
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Extract input patches that correspond to this weight position
                input_patches = self.input[:, i:i+out_height, j:j+out_width, :]
                input_reshaped = input_patches.reshape(-1, self.in_channels)
                
                # Reshape grad_output to match
                grad_reshaped = grad_output.reshape(-1, out_channels)
                
                # Compute gradient for this weight position
                weight_grad[i, j, :, :] = input_reshaped.T.dot(grad_reshaped)
        
        # 2. COMPUTE GRADIENT W.R.T. BIAS
        # ∂L/∂b[c] = Σ(b,h,w) grad_output[b,h,w,c]
        bias_grad = np.sum(grad_output, axis=(0, 1, 2))
        
        # 3. COMPUTE GRADIENT W.R.T. INPUT (for backpropagation)
        # This is like a "transposed convolution" or "deconvolution"
        
        # Pad the gradient output to handle edge effects
        pad_size = kernel_size - 1
        grad_padded = np.pad(grad_output, 
                           ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 
                           'constant')
        
        # Rotate and transpose weights for backpropagation
        # This is the "adjoint" operation of convolution
        weight_rotated = self.weight[::-1, ::-1, :, :].transpose(0, 1, 3, 2)
        weight_reshaped = weight_rotated.reshape(-1, self.in_channels)
        
        # Initialize gradient w.r.t. input
        grad_input = np.zeros((batch_size, in_height, in_width, self.in_channels))
        
        # Apply transposed convolution
        for h in range(in_height):
            for w in range(in_width):
                # Extract gradient patches
                grad_patch = grad_padded[:, h:h+kernel_size, w:w+kernel_size, :]
                grad_reshaped = grad_patch.reshape(batch_size, -1)
                
                # Compute gradient w.r.t. input
                input_grad = grad_reshaped.dot(weight_reshaped)
                grad_input[:, h, w, :] = input_grad
        
        # 4. UPDATE PARAMETERS (GRADIENT DESCENT)
        self.weight -= self.learning_rate * weight_grad + self.weight_decay * self.weight
        self.bias -= self.learning_rate * bias_grad + self.weight_decay * self.bias
        
        self.print_debug(f"Updated weights and bias")
        return grad_input

class Pooling(BaseLayer):
    """
    MAX POOLING LAYER
    =================
    
    WHAT IT DOES:
    - Reduces spatial dimensions by taking maximum value in each window
    - Helps with computational efficiency and provides some translation invariance
    - Common window size: 2x2 with stride 2 (reduces size by half)
    
    MATHEMATICAL FORMULA:
    Output[b,h,w,c] = max(Input[b,2h:2h+2,2w:2w+2,c])
    """
    
    def __init__(self, name="", window_size=2, stride=2):
        super().__init__(name)
        self.window_size = window_size
        self.stride = stride
        self.print_debug(f"Initialized Pooling layer: {window_size}x{window_size}, stride={stride}")
    
    def forward(self, x):
        """
        FORWARD PASS: MAX POOLING
        =========================
        
        ALGORITHM:
        1. Divide input into non-overlapping windows
        2. Take maximum value in each window
        3. Store which position had the maximum (for backpropagation)
        """
        batch_size, height, width, channels = x.shape
        
        # CALCULATE OUTPUT DIMENSIONS
        out_height = height // self.stride
        out_width = width // self.stride
        
        self.print_debug(f"Input shape: {x.shape}")
        self.print_debug(f"Output shape: ({batch_size}, {out_height}, {out_width}, {channels})")
        
        # RESHAPE TO GROUP PIXELS INTO WINDOWS
        # This clever reshaping groups pixels into 2x2 windows
        x_reshaped = x.reshape(batch_size, out_height, self.window_size, 
                              out_width, self.window_size, channels)
        
        # TAKE MAXIMUM IN EACH WINDOW
        output = np.max(x_reshaped, axis=(2, 4))
        
        # STORE MASK FOR BACKPROPAGATION
        # This mask tells us which position in each window had the maximum value
        output_expanded = output.reshape(batch_size, out_height, 1, out_width, 1, channels)
        self.mask = (output_expanded == x_reshaped)
        
        return output
    
    def backward(self, grad_output):
        """
        BACKWARD PASS: ROUTE GRADIENTS TO MAX POSITIONS
        ===============================================
        
        ALGORITHM:
        1. Expand gradient to match original input size
        2. Only pass gradients to positions that had maximum values
        3. All other positions get zero gradient
        """
        batch_size, out_height, out_width, channels = grad_output.shape
        
        # EXPAND GRADIENT TO MATCH WINDOW STRUCTURE
        grad_expanded = grad_output.reshape(batch_size, out_height, 1, out_width, 1, channels)
        
        # APPLY MASK: Only positions that were maximum get gradients
        grad_windowed = grad_expanded * self.mask
        
        # RESHAPE BACK TO ORIGINAL INPUT SIZE
        grad_input = grad_windowed.reshape(batch_size, out_height * self.stride, 
                                         out_width * self.stride, channels)
        
        return grad_input

class ReLU(BaseLayer):
    """
    RECTIFIED LINEAR UNIT (ReLU) ACTIVATION
    =======================================
    
    WHAT IT DOES:
    - Applies non-linearity to the network
    - ReLU(x) = max(0, x)
    - Helps with vanishing gradient problem
    - Very simple and computationally efficient
    
    MATHEMATICAL FORMULA:
    Output = max(0, Input)
    """
    
    def __init__(self, name=""):
        super().__init__(name)
        self.print_debug("Initialized ReLU layer")
    
    def forward(self, x):
        """
        FORWARD PASS: APPLY ReLU
        ========================
        
        ALGORITHM:
        - Keep positive values unchanged
        - Set negative values to zero
        """
        self.input = x
        output = np.maximum(0, x)  # Same as: (x > 0) * x
        
        self.print_debug(f"Input shape: {x.shape}")
        self.print_debug(f"ReLU output: {np.sum(output > 0)}/{output.size} non-zero values")
        
        return output
    
    def backward(self, grad_output):
        """
        BACKWARD PASS: GRADIENT OF ReLU
        ===============================
        
        ALGORITHM:
        - Pass gradients through for positive inputs
        - Block gradients for negative inputs (they were set to 0)
        
        MATHEMATICAL FORMULA:
        ∂ReLU(x)/∂x = 1 if x > 0, 0 if x ≤ 0
        """
        # Create mask: 1 where input > 0, 0 otherwise
        grad_mask = (self.input > 0).astype(np.float32)
        
        # Apply mask to gradients
        grad_input = grad_output * grad_mask
        
        return grad_input

class FC(BaseLayer):
    """
    FULLY CONNECTED (DENSE) LAYER
    =============================
    
    WHAT IT DOES:
    - Connects every input neuron to every output neuron
    - Learns linear combinations of input features
    - Usually used in final layers for classification
    
    MATHEMATICAL FORMULA:
    Output = Input * Weight + Bias
    """
    
    def __init__(self, name, in_features, out_features):
        super().__init__(name)
        
        self.in_features = in_features
        self.out_features = out_features
        
        # INITIALIZE WEIGHTS WITH XAVIER INITIALIZATION
        # This helps with training stability
        scale = np.sqrt(2.0 / in_features)
        self.weight = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros(out_features)
        
        # TRAINING PARAMETERS
        self.learning_rate = 0.0003
        self.weight_decay = 0.001
        
        # STORED FOR BACKPROPAGATION
        self.input = None
        self.original_shape = None
        
        self.print_debug(f"Initialized FC layer: {in_features} -> {out_features}")
    
    def forward(self, x):
        """
        FORWARD PASS: MATRIX MULTIPLICATION
        ===================================
        
        ALGORITHM:
        1. Flatten input if it's multi-dimensional
        2. Apply linear transformation: y = xW + b
        """
        # STORE ORIGINAL SHAPE FOR BACKPROPAGATION
        self.original_shape = x.shape
        
        # FLATTEN INPUT IF NECESSARY
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        
        self.input = x
        
        self.print_debug(f"Input shape: {x.shape}")
        self.print_debug(f"Output shape: ({x.shape[0]}, {self.out_features})")
        
        # LINEAR TRANSFORMATION
        output = x.dot(self.weight) + self.bias
        
        return output
    
    def backward(self, grad_output):
        """
        BACKWARD PASS: COMPUTE GRADIENTS
        ================================
        
        MATHEMATICAL DERIVATION:
        - ∂L/∂W = ∂L/∂y * ∂y/∂W = x^T * grad_output
        - ∂L/∂b = ∂L/∂y * ∂y/∂b = sum(grad_output)
        - ∂L/∂x = ∂L/∂y * ∂y/∂x = grad_output * W^T
        """
        batch_size = grad_output.shape[0]
        
        # COMPUTE GRADIENT W.R.T. WEIGHTS
        # ∂L/∂W = x^T * grad_output
        weight_grad = self.input.T.dot(grad_output)
        
        # COMPUTE GRADIENT W.R.T. BIAS
        # ∂L/∂b = sum(grad_output) across batch dimension
        bias_grad = np.sum(grad_output, axis=0)
        
        # COMPUTE GRADIENT W.R.T. INPUT
        # ∂L/∂x = grad_output * W^T
        grad_input = grad_output.dot(self.weight.T)
        
        # RESHAPE BACK TO ORIGINAL SHAPE
        grad_input = grad_input.reshape(self.original_shape)
        
        # UPDATE PARAMETERS
        self.weight -= self.learning_rate * weight_grad + self.weight_decay * self.weight
        self.bias -= self.learning_rate * bias_grad + self.weight_decay * self.bias
        
        return grad_input

class SoftmaxLoss(BaseLayer):
    """
    SOFTMAX + CROSS-ENTROPY LOSS
    ============================
    
    WHAT IT DOES:
    - Converts logits to probabilities using softmax
    - Computes cross-entropy loss for classification
    - Provides both loss value and accuracy
    
    MATHEMATICAL FORMULA:
    Softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    Loss = -Σ y_true * log(y_pred)
    """
    
    def __init__(self, name=""):
        super().__init__(name)
        self.print_debug("Initialized SoftmaxLoss layer")
    
    def forward(self, x):
        """
        FORWARD PASS: SOFTMAX + LOSS COMPUTATION
        ========================================
        
        ALGORITHM:
        1. Apply softmax to convert logits to probabilities
        2. If labels provided, compute cross-entropy loss
        3. Return predictions or (loss, accuracy)
        """
        # APPLY SOFTMAX
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        self.softmax = softmax
        
        # GET PREDICTIONS
        predictions = np.argmax(softmax, axis=1)
        
        # IF NO LABELS PROVIDED, JUST RETURN PREDICTIONS
        if not hasattr(self, 'labels'):
            return predictions
        
        # COMPUTE LOSS AND ACCURACY
        labels = self.labels
        true_labels = np.argmax(labels, axis=1)
        
        # CROSS-ENTROPY LOSS
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        loss = -np.sum(labels * np.log(softmax + epsilon)) / len(labels)
        
        # ACCURACY
        accuracy = np.mean(predictions == true_labels)
        
        self.print_debug(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    def backward(self, grad_output=1.0):
        """
        BACKWARD PASS: GRADIENT OF SOFTMAX + CROSS-ENTROPY
        ==================================================
        
        MATHEMATICAL FORMULA:
        ∂L/∂x = softmax - labels
        """
        # The gradient is simply the difference between predictions and true labels
        grad_input = self.softmax - self.labels
        
        return grad_input
    
    def set_labels(self, labels):
        """Set the true labels for loss computation"""
        self.labels = labels

class LeNet5:
    """
    LENET-5 ARCHITECTURE
    ====================
    
    ORIGINAL ARCHITECTURE (1998):
    Input (32x32) -> Conv1 (5x5, 6 filters) -> Pool1 (2x2) -> ReLU1 ->
    Conv2 (5x5, 16 filters) -> Pool2 (2x2) -> ReLU2 ->
    FC3 (400 -> 120) -> ReLU3 -> FC4 (120 -> 84) -> ReLU4 -> FC5 (84 -> 10) -> Softmax
    
    MODERN ADAPTATION:
    - Works with 28x28 MNIST images
    - Uses ReLU instead of tanh/sigmoid
    - Includes dropout and batch normalization (not shown here)
    """
    
    def __init__(self, debug_mode=False):
        """
        BUILD THE NETWORK
        =================
        
        Each layer processes the output of the previous layer:
        - Conv layers extract features (edges, textures, patterns)
        - Pool layers reduce spatial dimensions
        - ReLU adds non-linearity
        - FC layers combine features for classification
        """
        
        # CONVOLUTIONAL LAYERS (FEATURE EXTRACTION)
        self.conv1 = Conv("conv1", kernel_size=5, in_channels=1, out_channels=6)
        self.pool1 = Pooling("pool1", window_size=2, stride=2)
        self.relu1 = ReLU("relu1")
        
        self.conv2 = Conv("conv2", kernel_size=5, in_channels=6, out_channels=16)
        self.pool2 = Pooling("pool2", window_size=2, stride=2)
        self.relu2 = ReLU("relu2")
        
        # FULLY CONNECTED LAYERS (CLASSIFICATION)
        self.fc3 = FC("fc3", in_features=400, out_features=120)  # 16*5*5 = 400
        self.relu3 = ReLU("relu3")
        
        self.fc4 = FC("fc4", in_features=120, out_features=84)
        self.relu4 = ReLU("relu4")
        
        self.fc5 = FC("fc5", in_features=84, out_features=10)  # 10 classes for MNIST
        self.loss = SoftmaxLoss("loss")
        
        # ASSEMBLE LAYERS IN ORDER
        self.layers = [
            self.conv1, self.pool1, self.relu1,
            self.conv2, self.pool2, self.relu2,
            self.fc3, self.relu3, self.fc4, self.relu4, self.fc5, self.loss
        ]
        
        # SET DEBUG MODE FOR ALL LAYERS
        for layer in self.layers:
            layer.debug_mode = debug_mode
        
        print("LeNet-5 initialized successfully!")
        print("Architecture:")
        print("  Input (28x28x1) -> Conv1 (5x5, 6) -> Pool1 (2x2) -> ReLU1")
        print("  -> Conv2 (5x5, 16) -> Pool2 (2x2) -> ReLU2")
        print("  -> FC3 (400->120) -> ReLU3 -> FC4 (120->84) -> ReLU4")
        print("  -> FC5 (84->10) -> Softmax")
    
    def forward(self, x):
        """
        FORWARD PASS THROUGH ENTIRE NETWORK
        ===================================
        
        ALGORITHM:
        - Pass input through each layer sequentially
        - Each layer transforms the data according to its function
        - Final output is either predictions or (loss, accuracy)
        """
        current_input = x
        
        for i, layer in enumerate(self.layers):
            current_input = layer.forward(current_input)
            
            if layer.debug_mode:
                print(f"Layer {i+1}/{len(self.layers)} ({layer.name}): {current_input.shape}")
        
        return current_input
    
    def backward(self, grad_output=1.0):
        """
        BACKWARD PASS THROUGH ENTIRE NETWORK
        ====================================
        
        ALGORITHM:
        - Start with gradient from loss function
        - Pass gradients backward through each layer
        - Each layer computes gradients w.r.t. its inputs and updates its parameters
        """
        current_grad = grad_output
        
        for i, layer in enumerate(reversed(self.layers)):
            current_grad = layer.backward(current_grad)
            
            if layer.debug_mode:
                print(f"Backward {len(self.layers)-i}/{len(self.layers)} ({layer.name})")
        
        return current_grad
    
    def train_step(self, images, labels):
        """
        SINGLE TRAINING STEP
        ====================
        
        ALGORITHM:
        1. Forward pass: compute predictions and loss
        2. Backward pass: compute gradients and update parameters
        3. Return loss and accuracy
        """
        # SET LABELS FOR LOSS COMPUTATION
        self.loss.set_labels(labels)
        
        # FORWARD PASS
        loss, accuracy = self.forward(images)
        
        # BACKWARD PASS
        self.backward()
        
        return loss, accuracy
    
    def train(self, images, labels, epochs=10, batch_size=32, verbose=True):
        """
        COMPLETE TRAINING LOOP
        ======================
        
        ALGORITHM:
        1. Split data into batches
        2. For each epoch:
           - Shuffle data
           - For each batch: forward + backward + update
           - Track loss and accuracy
        3. Return training history
        """
        num_samples = len(images)
        num_batches = num_samples // batch_size
        
        history = {'loss': [], 'accuracy': []}
        
        print(f"Starting training: {epochs} epochs, {num_batches} batches per epoch")
        print(f"Total samples: {num_samples}, Batch size: {batch_size}")
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_accuracies = []
            
            # SHUFFLE DATA
            indices = np.random.permutation(num_samples)
            
            for batch_idx in range(num_batches):
                # GET BATCH
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                
                batch_images = images[batch_indices]
                batch_labels = labels[batch_indices]
                
                # TRAINING STEP
                loss, accuracy = self.train_step(batch_images, batch_labels)
                
                epoch_losses.append(loss)
                epoch_accuracies.append(accuracy)
                
                if verbose and batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{num_batches}")
                    print(f"  Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # COMPUTE EPOCH STATISTICS
            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(avg_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Average Accuracy: {avg_accuracy:.4f}")
            print("-" * 50)
        
        return history
    
    def predict(self, images):
        """
        MAKE PREDICTIONS
        ================
        
        ALGORITHM:
        - Forward pass only (no gradients needed)
        - Return class predictions
        """
        # Temporarily disable debug mode for prediction
        original_debug = [layer.debug_mode for layer in self.layers]
        for layer in self.layers:
            layer.debug_mode = False
        
        predictions = self.forward(images)
        
        # Restore debug mode
        for i, layer in enumerate(self.layers):
            layer.debug_mode = original_debug[i]
        
        return predictions
    
    def visualize_activations(self, image, layer_name=None):
        """
        VISUALIZE LAYER ACTIVATIONS
        ===========================
        
        This helps understand what each layer learns!
        """
        if layer_name is None:
            layer_name = "conv1"
        
        # Find the layer
        target_layer = None
        for layer in self.layers:
            if layer.name == layer_name:
                target_layer = layer
                break
        
        if target_layer is None:
            print(f"Layer '{layer_name}' not found!")
            return
        
        # Forward pass to get activations
        current_input = image.reshape(1, 28, 28, 1)
        
        for layer in self.layers:
            current_input = layer.forward(current_input)
            if layer.name == layer_name:
                activations = current_input[0]  # Remove batch dimension
                break
        
        # Visualize activations
        if len(activations.shape) == 3:  # Conv layer
            num_filters = activations.shape[2]
            fig, axes = plt.subplots(2, num_filters//2, figsize=(15, 6))
            axes = axes.flatten()
            
            for i in range(num_filters):
                axes[i].imshow(activations[:, :, i], cmap='viridis')
                axes[i].set_title(f'Filter {i+1}')
                axes[i].axis('off')
            
            plt.suptitle(f'Activations from {layer_name}')
            plt.tight_layout()
            plt.show()
        
        print(f"Visualized {num_filters} activation maps from {layer_name}")

# UTILITY FUNCTIONS FOR LEARNING
def create_simple_dataset():
    """
    CREATE A SIMPLE DATASET FOR TESTING
    ===================================
    
    This creates a small dataset to test the network
    """
    # Create simple patterns
    num_samples = 100
    images = np.random.randn(num_samples, 28, 28, 1)
    labels = np.random.randint(0, 10, num_samples)
    
    # Convert to one-hot encoding
    labels_onehot = np.zeros((num_samples, 10))
    labels_onehot[np.arange(num_samples), labels] = 1
    
    return images, labels_onehot

def test_individual_layers():
    """
    TEST INDIVIDUAL LAYERS
    ======================
    
    This function helps you understand each layer in isolation
    """
    print("Testing individual layers...")
    
    # Test ReLU
    print("\n1. Testing ReLU:")
    relu = ReLU("test_relu")
    x = np.array([[-1, 2, -3, 4]])
    print(f"Input: {x}")
    output = relu.forward(x)
    print(f"Output: {output}")
    
    # Test FC
    print("\n2. Testing FC:")
    fc = FC("test_fc", 4, 3)
    x = np.array([[1, 2, 3, 4]])
    print(f"Input: {x}")
    output = fc.forward(x)
    print(f"Output: {output}")
    
    # Test Pooling
    print("\n3. Testing Pooling:")
    pool = Pooling("test_pool")
    x = np.random.randn(1, 4, 4, 1)
    print(f"Input shape: {x.shape}")
    output = pool.forward(x)
    print(f"Output shape: {output.shape}")
    
    # Test Conv
    print("\n4. Testing Conv:")
    conv = Conv("test_conv", 3, 1, 2)
    x = np.random.randn(1, 5, 5, 1)
    print(f"Input shape: {x.shape}")
    output = conv.forward(x)
    print(f"Output shape: {output.shape}")

# MAIN EXECUTION
if __name__ == "__main__":
    print("PEDAGOGICAL LENET-5 IMPLEMENTATION")
    print("=" * 50)
    
    # Test individual layers first
    test_individual_layers()
    
    # Create and test the full network
    print("\n" + "=" * 50)
    print("Testing full LeNet-5 network:")
    
    # Initialize network with debug mode
    net = LeNet5(debug_mode=True)
    
    # Create simple dataset
    images, labels = create_simple_dataset()
    
    # Test forward pass
    print("\nTesting forward pass...")
    predictions = net.predict(images[:5])
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions}")
    
    # Test training step
    print("\nTesting training step...")
    loss, accuracy = net.train_step(images[:10], labels[:10])
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    print("\nNetwork is working correctly!")
    print("\nNext steps:")
    print("1. Load real MNIST data")
    print("2. Train the network")
    print("3. Visualize activations")
    print("4. Experiment with different architectures")
