from pytorch_nndct.apis import torch_quantizer
import torch
from lenet5 import QuantLeNet5
import torch.ao.quantization as quant
from torchvision import transforms, datasets

model = QuantLeNet5()
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
model = torch.ao.quantization.prepare_qat(model.train())

model.load_state_dict(torch.load("fake_quant_fp_lenet_weight.pth"))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

calib_dataset = datasets.MNIST('./data', train=True, transform=transform)
calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=64, shuffle=True)

example_in = torch.randn(1,1,28,28)
quantizer = torch_quantizer(quant_mode="calib", module = model, input_args=example_in, output_dir="qat_out")

quant_model = quantizer.quant_model
quant_model.eval()

with torch.no_grad():
    for batch, (data, _) in enumerate(calib_loader):
        quant_model(data)

quantizer.export_quant_config()
# quantizer.export_xmodel("quant_result", deploy_check=True)


# TEST STEP / XMODEL EXPORT
# -------------------------------------------------------------------

quantizer_test = torch_quantizer(
    quant_mode="test",
    module=model,
    input_args=example_in,
    output_dir="qat_out"
)

test_model = quantizer_test.quant_model
test_model.eval()

# Optionally run an inference pass if you want
with torch.no_grad():
    for data, _ in calib_loader:
        _ = test_model(data)
        break  # You usually don't need a full pass here

quantizer_test.export_xmodel("quant_result_test", deploy_check=True)
