import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score


# Resnet block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Data load
class CustomImageFolder(ImageFolder):
    def __init__(self, root, args, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.args = args

    def __getitem__(self, index):
        # Init
        path, target = self.samples[index]
        emotions2id = self.args.emotions2id

        # Label
        target = -1
        for key, value in emotions2id.items():
            if key in path:
                target = value
                break
        
        # Transform
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


# Metrics
def calculate_metrics(predictions, labels, num_classes):
    precision = precision_score(labels, predictions, average=None, labels=num_classes)
    recall = recall_score(labels, predictions, average=None, labels=num_classes)
    f1 = f1_score(labels, predictions, average=None, labels=num_classes)
    return precision, recall, f1


# Data init
def dataInit(args):
    # data augmentation
    data_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # train && eval
    train_data = CustomImageFolder(root=args.train_dataset_path, args=args, transform=data_transform)
    test_dataset = CustomImageFolder(root=args.test_dataset_path, args=args, transform=data_transform)

    train_size = int(args.train_eval_ratio * len(train_data))
    eval_size = len(train_data) - train_size
    train_dataset, eval_dataset = random_split(train_data, [train_size, eval_size])

    # loader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, eval_dataloader, test_dataloader



def basicInit(emotions_label=['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']):
    # parameters
    parser = argparse.ArgumentParser(description='Hyper Parameters')

    parser.add_argument('--train_dataset_path', type=str, default='dataset/mid_hw_emotion_recognition/train/', help='train dataset')
    parser.add_argument('--test_dataset_path', type=str, default='dataset/mid_hw_emotion_recognition/test/', help='test dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch')
    parser.add_argument('--train_eval_ratio', type=float, default=0.8, help='train dataset -> train + eval data')

    id2emotions = dict(enumerate(emotions_label))
    emotions2id = {v:k for k,v in id2emotions.items()}
    parser.add_argument('--emotions_label', type=list, default=emotions_label, help='emotion labels, preserved in the code')
    parser.add_argument('--id2emotions', type=dict, default=id2emotions, help='id to str')
    parser.add_argument('--emotions2id', type=dict, default=emotions2id, help='str to id')

    parser.add_argument('--layer_repeats', type=list, default=[2, 2, 2, 2], help='ResNet basic blocks')
    parser.add_argument('--num_classes', type=int, default=len(emotions_label), help='5 emotion labels')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='momentum')
    parser.add_argument('--num_epoches', type=int, default=10, help='epoches')

    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/', help='model eval checkpoints')

    args = parser.parse_args()

    # model
    model = ResNet(BasicBlock, args.layer_repeats, args.num_classes)
    return args, model
