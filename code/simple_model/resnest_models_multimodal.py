import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)  # Add dropout layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout after activation

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# CNN Module for processing RGB images
class RGBEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(RGBEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, output_dim)
    
    def forward(self, x):
        # Input x shape: [batch_size, frames, height, width, channels]
        # Reshape to process each frame separately
        batch_size, frames, height, width, channels = x.shape
        x = x.permute(0, 1, 4, 2, 3)  # [batch, frames, channels, height, width]
        
        # Reshape to process all frames as batch dimension
        x = x.reshape(batch_size * frames, channels, height, width)
        
        # Apply CNN layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # Reshape back to [batch, frames, features]
        x = x.reshape(batch_size, frames, -1)
        
        # Transpose to get [batch, features, frames] for 1D convs
        x = x.transpose(1, 2)
        
        return x

class MultiModalResNet(nn.Module):
    def __init__(self, block, layers, state_channels=9, fusion_type='concat', 
                 rgb_feature_dim=128, zero_init_residual=False, dropout_rate=0.5):
        super(MultiModalResNet, self).__init__()
        self.fusion_type = fusion_type
        self.dropout_rate = dropout_rate
        
        # RGB encoders
        self.rgb_encoder1 = RGBEncoder(output_dim=rgb_feature_dim)
        self.rgb_encoder2 = RGBEncoder(output_dim=rgb_feature_dim)
        
        # Set up channels for state processing
        self.state_channels = state_channels
        
        # Calculate fused input channels based on fusion type
        if fusion_type == 'concat':
            # Concatenate features from states and RGB
            self.in_channels = 64
            self.fused_channels = state_channels + rgb_feature_dim * 2
            
            # Fusion layer to reduce concatenated dimensions
            self.fusion_layer = nn.Conv1d(self.fused_channels, 64, 
                                         kernel_size=1, stride=1, bias=False)
            self.fusion_bn = nn.BatchNorm1d(64)
            self.fusion_relu = nn.ReLU(inplace=True)
        else:  # 'sum' or other fusion methods
            self.in_channels = 64
            # For sum fusion, we'll project each modality to the same dimension
            self.state_proj = nn.Conv1d(state_channels, 64, kernel_size=1, bias=False)
            self.rgb1_proj = nn.Conv1d(rgb_feature_dim, 64, kernel_size=1, bias=False)
            self.rgb2_proj = nn.Conv1d(rgb_feature_dim, 64, kernel_size=1, bias=False)
            self.fusion_bn = nn.BatchNorm1d(64)
            self.fusion_relu = nn.ReLU(inplace=True)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * (block.expansion if hasattr(block, 'expansion') else 1), 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * (block.expansion if hasattr(block, 'expansion') else 1):
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * (block.expansion if hasattr(block, 'expansion') else 1),
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * (block.expansion if hasattr(block, 'expansion') else 1)),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * (block.expansion if hasattr(block, 'expansion') else 1)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, states, rgb1, rgb2):
        # Process RGB inputs through their encoders
        rgb1_features = self.rgb_encoder1(rgb1)
        rgb2_features = self.rgb_encoder2(rgb2)
        
        # Ensure all features have the same sequence length dimension for concatenation
        # Assuming the sequence dimension is the last dimension (dim=2)
        target_seq_len = states.shape[2]
        
        # Resize RGB features if necessary
        if rgb1_features.shape[2] != target_seq_len:
            rgb1_features = torch.nn.functional.interpolate(
                rgb1_features, 
                size=target_seq_len,
                mode='linear', 
                align_corners=False
            )
            
        if rgb2_features.shape[2] != target_seq_len:
            rgb2_features = torch.nn.functional.interpolate(
                rgb2_features, 
                size=target_seq_len,
                mode='linear', 
                align_corners=False
            )
        
        # Fusion of different modalities
        if self.fusion_type == 'concat':
            # Concatenate along channel dimension
            x = torch.cat([states, rgb1_features, rgb2_features], dim=1)
            
            # Apply fusion layer to reduce dimensions
            x = self.fusion_layer(x)
            x = self.fusion_bn(x)
            x = self.fusion_relu(x)
        else:  # 'sum' fusion
            # Project each modality to the same dimension
            states_proj = self.state_proj(states)
            rgb1_proj = self.rgb1_proj(rgb1_features)
            rgb2_proj = self.rgb2_proj(rgb2_features)
            
            # Sum the projections
            x = states_proj + rgb1_proj + rgb2_proj
            x = self.fusion_bn(x)
            x = self.fusion_relu(x)
        
        # Apply ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

def multimodal_resnet18(state_channels=9, fusion_type='concat', rgb_feature_dim=128, dropout_rate=0.5, **kwargs):
    return MultiModalResNet(ResidualBlock, [2, 2, 2, 2], 
                           state_channels=state_channels, 
                           fusion_type=fusion_type,
                           rgb_feature_dim=rgb_feature_dim,
                           dropout_rate=dropout_rate, **kwargs)

def multimodal_resnet34(state_channels=9, fusion_type='concat', rgb_feature_dim=128, dropout_rate=0.5, **kwargs):
    return MultiModalResNet(ResidualBlock, [3, 4, 6, 3], 
                           state_channels=state_channels, 
                           fusion_type=fusion_type,
                           rgb_feature_dim=rgb_feature_dim,
                           dropout_rate=dropout_rate, **kwargs)

def multimodal_resnet50(state_channels=9, fusion_type='concat', rgb_feature_dim=128, dropout_rate=0.5, **kwargs):
    return MultiModalResNet(BottleneckBlock, [3, 4, 6, 3], 
                           state_channels=state_channels, 
                           fusion_type=fusion_type,
                           rgb_feature_dim=rgb_feature_dim,
                           dropout_rate=dropout_rate, **kwargs)

def multimodal_resnet101(state_channels=9, fusion_type='concat', rgb_feature_dim=128, dropout_rate=0.5, **kwargs):
    return MultiModalResNet(BottleneckBlock, [3, 4, 23, 3], 
                           state_channels=state_channels, 
                           fusion_type=fusion_type,
                           rgb_feature_dim=rgb_feature_dim,
                           dropout_rate=dropout_rate, **kwargs)

def multimodal_resnet152(state_channels=9, fusion_type='concat', rgb_feature_dim=128, dropout_rate=0.5, **kwargs):
    return MultiModalResNet(BottleneckBlock, [3, 8, 36, 3], 
                           state_channels=state_channels, 
                           fusion_type=fusion_type,
                           rgb_feature_dim=rgb_feature_dim,
                           dropout_rate=dropout_rate, **kwargs)