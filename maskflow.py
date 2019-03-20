#!/usr/bin/env python
import os
import sys
import math
import numpy as np
import cv2
import torch
import cvbase as cvb
import skimage.io
import matplotlib.pyplot as plt

# Import Mask RCNN
ROOT_DIR_MASK = os.path.abspath("./Mask_RCNN")
sys.path.append(ROOT_DIR_MASK)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR_MASK, "samples/coco/"))  # To find local version
import coco
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR_MASK, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR_MASK, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# Import liteflownet
ROOT_DIR_LITE = os.path.abspath("./liteflownet")
sys.path.append(ROOT_DIR_LITE)
try:
    from correlation import correlation  # the custom cost volume layer
except:
    sys.path.append('./correlation')
    import correlation  # you should consider upgrading python

# pretrained model for flow prdiction
flow_model = 'default'


assert (int(str('').join(torch.__version__.split('.')[0:3])) >= 41)  # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance

torch.cuda.device(1)  # change this if you have a multiple graphics cards and you want to utilize them

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance


# Mask RCNN
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def maskPeople(image, flow_image):
    # mask people in the flow image
    # input:
    # image -- the first image of the image pair for flow calculation
    # flow_image -- flow field in an rgb image
    # output:
    # num_people -- number of people in the image
    # image_batch -- image(s) of mask of each person in th flow image

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]

    # print(np.shape(r['masks']))
    num_people = np.sum(r['class_ids'] == 1)  # 1 is the id index for person

    image_batch = np.zeros(
        (num_people, np.shape(flow_image)[0], np.shape(flow_image)[1], 3),
        dtype=np.uint8)
    j = 0
    for i in range(len(r['class_ids'])):
        if r['class_ids'][i] == 1:
            image_batch[j] = flow_image
            image_batch[j, :, :, 0] = np.multiply(image_batch[j, :, :, 0],
                                                  r['masks'][:, :, i])
            image_batch[j, :, :, 1] = np.multiply(image_batch[j, :, :, 1],
                                                  r['masks'][:, :, i])
            image_batch[j, :, :, 2] = np.multiply(image_batch[j, :, :, 2],
                                                  r['masks'][:, :, i])
            j += 1

    return num_people, image_batch


Backward_tensorGrid = {}


def Backward(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
            1, 1, 1, tensorFlow.size(3)).expand(
                tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
            1, 1, tensorFlow.size(2), 1).expand(
                tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat(
            [tensorHorizontal, tensorVertical], 1).cuda()
    # end

    tensorFlow = torch.cat([
        tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
        tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)
    ], 1)

    return torch.nn.functional.grid_sample(
        input=tensorInput,
        grid=(
            Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(
                0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros')


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        class Features(torch.nn.Module):
            def __init__(self):
                super(Features, self).__init__()

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=3,
                        out_channels=32,
                        kernel_size=7,
                        stride=1,
                        padding=3),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=96,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=192,
                        kernel_size=3,
                        stride=2,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

            # end

            def forward(self, tensorInput):
                tensorOne = self.moduleOne(tensorInput)
                tensorTwo = self.moduleTwo(tensorOne)
                tensorThr = self.moduleThr(tensorTwo)
                tensorFou = self.moduleFou(tensorThr)
                tensorFiv = self.moduleFiv(tensorFou)
                tensorSix = self.moduleSix(tensorFiv)

                return [
                    tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv,
                    tensorSix
                ]

            # end

        # end

        class Matching(torch.nn.Module):
            def __init__(self, intLevel):
                super(Matching, self).__init__()

                self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25,
                                    0.625][intLevel]

                if intLevel != 2:
                    self.moduleFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.moduleFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels=32,
                            out_channels=64,
                            kernel_size=1,
                            stride=1,
                            padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                # end

                if intLevel == 6:
                    self.moduleUpflow = None

                elif intLevel != 6:
                    self.moduleUpflow = torch.nn.ConvTranspose2d(
                        in_channels=2,
                        out_channels=2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                        groups=2)

                # end

                if intLevel >= 4:
                    self.moduleUpcorr = None

                elif intLevel < 4:
                    self.moduleUpcorr = torch.nn.ConvTranspose2d(
                        in_channels=49,
                        out_channels=49,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                        groups=49)

                # end

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=49,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=2,
                        kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel],
                        stride=1,
                        padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))

            # end

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst,
                        tensorFeaturesSecond, tensorFlow):
                tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
                tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)

                if tensorFlow is not None:
                    tensorFlow = self.moduleUpflow(tensorFlow)
                # end

                if tensorFlow is not None:
                    tensorFeaturesSecond = Backward(
                        tensorInput=tensorFeaturesSecond,
                        tensorFlow=tensorFlow * self.dblBackward)
                # end

                if self.moduleUpcorr is None:
                    tensorCorrelation = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(
                            tensorFirst=tensorFeaturesFirst,
                            tensorSecond=tensorFeaturesSecond,
                            intStride=1),
                        negative_slope=0.1,
                        inplace=False)

                elif self.moduleUpcorr is not None:
                    tensorCorrelation = self.moduleUpcorr(
                        torch.nn.functional.leaky_relu(
                            input=correlation.FunctionCorrelation(
                                tensorFirst=tensorFeaturesFirst,
                                tensorSecond=tensorFeaturesSecond,
                                intStride=2),
                            negative_slope=0.1,
                            inplace=False))

                # end

                return (tensorFlow if tensorFlow is not None else
                        0.0) + self.moduleMain(tensorCorrelation)

            # end

        # end

        class Subpixel(torch.nn.Module):
            def __init__(self, intLevel):
                super(Subpixel, self).__init__()

                self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25,
                                    0.625][intLevel]

                if intLevel != 2:
                    self.moduleFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.moduleFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels=32,
                            out_channels=64,
                            kernel_size=1,
                            stride=1,
                            padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                # end

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=[0, 0, 130, 130, 194, 258, 386][intLevel],
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=2,
                        kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel],
                        stride=1,
                        padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))

            # end

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst,
                        tensorFeaturesSecond, tensorFlow):
                tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
                tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)

                if tensorFlow is not None:
                    tensorFeaturesSecond = Backward(
                        tensorInput=tensorFeaturesSecond,
                        tensorFlow=tensorFlow * self.dblBackward)
                # end

                return (tensorFlow
                        if tensorFlow is not None else 0.0) + self.moduleMain(
                            torch.cat([
                                tensorFeaturesFirst, tensorFeaturesSecond,
                                tensorFlow
                            ], 1))

            # end

        # end

        class Regularization(torch.nn.Module):
            def __init__(self, intLevel):
                super(Regularization, self).__init__()

                self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25,
                                    0.625][intLevel]

                self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]

                if intLevel >= 5:
                    self.moduleFeat = torch.nn.Sequential()

                elif intLevel < 5:
                    self.moduleFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels=[0, 0, 32, 64, 96, 128, 192][intLevel],
                            out_channels=128,
                            kernel_size=1,
                            stride=1,
                            padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                # end

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=[0, 0, 131, 131, 131, 131, 195][intLevel],
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

                if intLevel >= 5:
                    self.moduleDist = torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels=32,
                            out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                            kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel],
                            stride=1,
                            padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))

                elif intLevel < 5:
                    self.moduleDist = torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels=32,
                            out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                            kernel_size=([0, 0, 7, 5, 5, 3, 3][intLevel], 1),
                            stride=1,
                            padding=([0, 0, 3, 2, 2, 1, 1][intLevel], 0)),
                        torch.nn.Conv2d(
                            in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                            out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                            kernel_size=(1, [0, 0, 7, 5, 5, 3, 3][intLevel]),
                            stride=1,
                            padding=(0, [0, 0, 3, 2, 2, 1, 1][intLevel])))

                # end

                self.moduleScaleX = torch.nn.Conv2d(
                    in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.moduleScaleY = torch.nn.Conv2d(
                    in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)

            # eny

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst,
                        tensorFeaturesSecond, tensorFlow):
                tensorDifference = (tensorFirst - Backward(
                    tensorInput=tensorSecond,
                    tensorFlow=tensorFlow * self.dblBackward)).pow(2.0).sum(
                        1, True).sqrt()

                tensorDist = self.moduleDist(
                    self.moduleMain(
                        torch.cat([
                            tensorDifference, tensorFlow - tensorFlow.view(
                                tensorFlow.size(0), 2, -1).mean(2, True).view(
                                    tensorFlow.size(0), 2, 1, 1),
                            self.moduleFeat(tensorFeaturesFirst)
                        ], 1)))
                tensorDist = tensorDist.pow(2.0).neg()
                tensorDist = (tensorDist - tensorDist.max(1, True)[0]).exp()

                tensorDivisor = tensorDist.sum(1, True).reciprocal()

                tensorScaleX = self.moduleScaleX(
                    tensorDist * torch.nn.functional.unfold(
                        input=tensorFlow[:, 0:1, :, :],
                        kernel_size=self.intUnfold,
                        stride=1,
                        padding=int(
                            (self.intUnfold - 1) / 2)).view_as(tensorDist)
                ) * tensorDivisor
                tensorScaleY = self.moduleScaleY(
                    tensorDist * torch.nn.functional.unfold(
                        input=tensorFlow[:, 1:2, :, :],
                        kernel_size=self.intUnfold,
                        stride=1,
                        padding=int(
                            (self.intUnfold - 1) / 2)).view_as(tensorDist)
                ) * tensorDivisor

                return torch.cat([tensorScaleX, tensorScaleY], 1)

            # end

        # end

        self.moduleFeatures = Features()
        self.moduleMatching = torch.nn.ModuleList(
            [Matching(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.moduleSubpixel = torch.nn.ModuleList(
            [Subpixel(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.moduleRegularization = torch.nn.ModuleList(
            [Regularization(intLevel) for intLevel in [2, 3, 4, 5, 6]])

        self.load_state_dict(
            torch.load('./liteflownet/network-' + flow_model + '.pytorch'))

    # end


    def forward(self, tensorFirst, tensorSecond):
        tensorFirst[:, 0, :, :] = tensorFirst[:, 0, :, :] - 0.411618
        tensorFirst[:, 1, :, :] = tensorFirst[:, 1, :, :] - 0.434631
        tensorFirst[:, 2, :, :] = tensorFirst[:, 2, :, :] - 0.454253

        tensorSecond[:, 0, :, :] = tensorSecond[:, 0, :, :] - 0.410782
        tensorSecond[:, 1, :, :] = tensorSecond[:, 1, :, :] - 0.433645
        tensorSecond[:, 2, :, :] = tensorSecond[:, 2, :, :] - 0.452793

        tensorFeaturesFirst = self.moduleFeatures(tensorFirst)
        tensorFeaturesSecond = self.moduleFeatures(tensorSecond)

        tensorFirst = [tensorFirst]
        tensorSecond = [tensorSecond]

        for intLevel in [1, 2, 3, 4, 5]:
            tensorFirst.append(
                torch.nn.functional.interpolate(
                    input=tensorFirst[-1],
                    size=(tensorFeaturesFirst[intLevel].size(2),
                          tensorFeaturesFirst[intLevel].size(3)),
                    mode='bilinear',
                    align_corners=False))
            tensorSecond.append(
                torch.nn.functional.interpolate(
                    input=tensorSecond[-1],
                    size=(tensorFeaturesSecond[intLevel].size(2),
                          tensorFeaturesSecond[intLevel].size(3)),
                    mode='bilinear',
                    align_corners=False))
        # end

        tensorFlow = None

        for intLevel in [-1, -2, -3, -4, -5]:
            tensorFlow = self.moduleMatching[intLevel](
                tensorFirst[intLevel], tensorSecond[intLevel],
                tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel],
                tensorFlow)
            tensorFlow = self.moduleSubpixel[intLevel](
                tensorFirst[intLevel], tensorSecond[intLevel],
                tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel],
                tensorFlow)
            tensorFlow = self.moduleRegularization[intLevel](
                tensorFirst[intLevel], tensorSecond[intLevel],
                tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel],
                tensorFlow)

        return tensorFlow * 20.0


moduleNetwork = Network().cuda().eval()


def estimate(tensorFirst, tensorSecond):
    assert (tensorFirst.size(1) == tensorSecond.size(1))
    assert (tensorFirst.size(2) == tensorSecond.size(2))

    intWidth = tensorFirst.size(2)
    intHeight = tensorFirst.size(1)

    assert (
        intWidth == 1024
    )  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert (
        intHeight == 436
    )  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight,
                                                      intWidth)
    tensorPreprocessedSecond = tensorSecond.cuda().view(
        1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tensorPreprocessedFirst = torch.nn.functional.interpolate(
        input=tensorPreprocessedFirst,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode='bilinear',
        align_corners=False)
    tensorPreprocessedSecond = torch.nn.functional.interpolate(
        input=tensorPreprocessedSecond,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode='bilinear',
        align_corners=False)

    tensorFlow = torch.nn.functional.interpolate(
        input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond),
        size=(intHeight, intWidth),
        mode='bilinear',
        align_corners=False)

    tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
    return tensorFlow[0, :, :, :].cpu()


def resize(img):
    # resize input pictures so that they fit into the trained model

    DIM = (1024, 436)  # the dimension of the images in liteflownet model
    if (img.shape[0], img.shape[0]) != DIM:
        img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)
    return img


def resize_back(img):
    DIM = (4032, 3024)
    if (img.shape[0], img.shape[0]) != DIM:
        img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)
    return img


def calculateFlow(image1, image2):
    resized_image1 = resize(image1)
    resized_image2 = resize(image2)
    tensor1 = torch.FloatTensor(resized_image1[:, :, ::-1].transpose(
        2, 0, 1).astype(np.float32) * (1.0 / 255.0))
    tensor2 = torch.FloatTensor(resized_image2[:, :, ::-1].transpose(
        2, 0, 1).astype(np.float32) * (1.0 / 255.0))
    tensorOutput = estimate(tensor1, tensor2)
    flow = np.array(tensorOutput.numpy().transpose(1, 2, 0), np.float32)
    return flow


def readFlow(name):
    # read .flo file to an array

    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:, :, 0:2]
    f = open(name, 'rb')
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')
    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)



def flow2rgb(flow):
    # convert flow field to an rgb image
    flow_image = cvb.flow2rgb(flow)
    flow_image *= 255
    flow_image = resize_back(flow_image)
    flow_image = np.array(flow_image,dtype = np.uint8)
    return flow_image


def maskFlow(first_image, second_image):
    # returns
    flow = calculateFlow(first_image, second_image)
    flow_image = flow2rgb(flow)
    num_people, color_images = maskPeople(first_image, flow_image)
    return num_people, color_images


if __name__ == "__main__":
    print('here is 77')

    first_image = skimage.io.imread("./data/IMG_2239.JPG")
    second_image = skimage.io.imread("./data/IMG_2240.JPG")

    num_people, color_images = maskFlow(first_image, second_image)
    print(str(num_people)+"detected")

    for i in range(num_people):
        # plt.imshow(color_images[0])
        cv2.imwrite("maskedflow"+str(i)+".png", color_images[i])