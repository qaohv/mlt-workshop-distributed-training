import torch

from main import Resnet50Classifier

if __name__ == "__main__":
    net = Resnet50Classifier().to("cuda:0")
    criterion = torch.nn.NLLLoss()

    image = torch.randn((1, 3, 224, 224))
    label = torch.LongTensor([10])

    output = net(image.to("cuda:0")).cpu()
    loss = criterion(output, label)
    loss.backward()

    total_nn_size = 0
    for name, p in list(net.named_parameters()):
        layer_size = (p.grad.element_size() * torch.prod(torch.LongTensor(list(p.grad.size())))).numpy()
        total_nn_size += layer_size

    print(f"NN size in mbytes: {total_nn_size * 1e-6}")
