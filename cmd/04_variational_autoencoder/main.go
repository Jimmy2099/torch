package main

// VAE(
//
//	(encoder_conv): Sequential(
//	  (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
//	  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (2): ReLU(inplace=True)
//	  (3): Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
//	  (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (5): ReLU(inplace=True)
//	  (6): Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
//	  (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (8): ReLU(inplace=True)
//	  (9): Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
//	  (10): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (11): ReLU(inplace=True)
//	)
//	(flatten): Flatten(start_dim=1, end_dim=-1)
//	(fc_mu): Linear(in_features=8192, out_features=64, bias=True)
//	(fc_logvar): Linear(in_features=8192, out_features=64, bias=True)
//	(decoder_fc): Linear(in_features=64, out_features=8192, bias=True)
//	(decoder_conv): Sequential(
//	  (0): ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
//	  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (2): ReLU(inplace=True)
//	  (3): ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
//	  (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (5): ReLU(inplace=True)
//	  (6): ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
//	  (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
//	  (8): ReLU(inplace=True)
//	  (9): ConvTranspose2d(64, 3, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
//	  (10): Tanh()
//	)
//
// )
// Layer: encoder_conv.0.weight, Shape: torch.Size([64, 3, 5, 5]), dim: 4
// Layer: encoder_conv.0.bias, Shape: torch.Size([64]), dim: 1
// Layer: encoder_conv.1.weight, Shape: torch.Size([64]), dim: 1
// Layer: encoder_conv.1.bias, Shape: torch.Size([64]), dim: 1
// Layer: encoder_conv.3.weight, Shape: torch.Size([128, 64, 5, 5]), dim: 4
// Layer: encoder_conv.3.bias, Shape: torch.Size([128]), dim: 1
// Layer: encoder_conv.4.weight, Shape: torch.Size([128]), dim: 1
// Layer: encoder_conv.4.bias, Shape: torch.Size([128]), dim: 1
// Layer: encoder_conv.6.weight, Shape: torch.Size([256, 128, 5, 5]), dim: 4
// Layer: encoder_conv.6.bias, Shape: torch.Size([256]), dim: 1
// Layer: encoder_conv.7.weight, Shape: torch.Size([256]), dim: 1
// Layer: encoder_conv.7.bias, Shape: torch.Size([256]), dim: 1
// Layer: encoder_conv.9.weight, Shape: torch.Size([512, 256, 5, 5]), dim: 4
// Layer: encoder_conv.9.bias, Shape: torch.Size([512]), dim: 1
// Layer: encoder_conv.10.weight, Shape: torch.Size([512]), dim: 1
// Layer: encoder_conv.10.bias, Shape: torch.Size([512]), dim: 1
// Layer: fc_mu.weight, Shape: torch.Size([64, 8192]), dim: 2
// Layer: fc_mu.bias, Shape: torch.Size([64]), dim: 1
// Layer: fc_logvar.weight, Shape: torch.Size([64, 8192]), dim: 2
// Layer: fc_logvar.bias, Shape: torch.Size([64]), dim: 1
// Layer: decoder_fc.weight, Shape: torch.Size([8192, 64]), dim: 2
// Layer: decoder_fc.bias, Shape: torch.Size([8192]), dim: 1
// Layer: decoder_conv.0.weight, Shape: torch.Size([512, 256, 5, 5]), dim: 4
// Layer: decoder_conv.0.bias, Shape: torch.Size([256]), dim: 1
// Layer: decoder_conv.1.weight, Shape: torch.Size([256]), dim: 1
// Layer: decoder_conv.1.bias, Shape: torch.Size([256]), dim: 1
// Layer: decoder_conv.3.weight, Shape: torch.Size([256, 128, 5, 5]), dim: 4
// Layer: decoder_conv.3.bias, Shape: torch.Size([128]), dim: 1
// Layer: decoder_conv.4.weight, Shape: torch.Size([128]), dim: 1
// Layer: decoder_conv.4.bias, Shape: torch.Size([128]), dim: 1
// Layer: decoder_conv.6.weight, Shape: torch.Size([128, 64, 5, 5]), dim: 4
// Layer: decoder_conv.6.bias, Shape: torch.Size([64]), dim: 1
// Layer: decoder_conv.7.weight, Shape: torch.Size([64]), dim: 1
// Layer: decoder_conv.7.bias, Shape: torch.Size([64]), dim: 1
// Layer: decoder_conv.9.weight, Shape: torch.Size([64, 3, 5, 5]), dim: 4
// Layer: decoder_conv.9.bias, Shape: torch.Size([3]), dim: 1

// TODO
func main() {

}
