import torchvision
from torchvision.transforms import transforms

IMAGE_DATASETS = [
    'LSUN', 'LSUNClass',
    'ImageFolder', 'DatasetFolder', 'FakeData',
    'CocoCaptions', 'CocoDetection',
    'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'QMNIST',
    'MNIST', 'KMNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
    'Omniglot', 'SBU', 'Flickr8k', 'Flickr30k',
    'VOCSegmentation', 'VOCDetection', 'Cityscapes', 'ImageNet',
    'Caltech101', 'Caltech256', 'CelebA', 'WIDERFace', 'SBDataset',
    'VisionDataset', 'USPS', 'Kinetics400', 'HMDB51', 'UCF101',
    'Places365', 'TinyImageNet'
]

if __name__ == '__main__':
    out = ""
    for task in IMAGE_DATASETS:
        f = open(f"./{task.lower()}.yaml", "w")
        f.write(r"""defaults:
      - _self_
    
suite: classify
train_steps: 200000
stddev_schedule: 'linear(1.0,0.1,100000)'
frame_stack: null
action_repeat: null
nstep: 0
evaluate_per_steps: 1000
evaluate_episodes: 1
learn_per_steps: 1
seed_steps: 50
explore_steps: 0
log_per_episodes: 10
offline: true
task_name: {}""".format(task))
        f.close()
        out += ' "' + task.lower() + '"'
    print(out)
