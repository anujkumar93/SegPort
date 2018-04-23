import torch.nn as nn


class FCN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(p=0.5, inplace=True)

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(p=0.5, inplace=True)

        self.score = nn.Conv2d(4096, 1, kernel_size=1)

        self.score_deconv1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)
        self.score_pool4 = nn.Conv2d(512, 1, kernel_size=1)

        self.score_deconv2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, bias=False)
        self.score_pool3 = nn.Conv2d(256, 1, kernel_size=1)

        self.score_deconv3 = nn.ConvTranspose2d(1, 1, kernel_size=16, stride=8, bias=False)

    def forward(self, batch):
        output_pool1 = self.pool1(self.relu1_2(self.conv1_2(
                                  self.relu1_1(self.conv1_1(batch)))))
        output_pool2 = self.pool2(self.relu2_2(self.conv2_2(
                                  self.relu2_1(self.conv2_1(output_pool1)))))
        output_pool3 = self.pool3(self.relu3_3(self.conv3_3(
                                  self.relu3_2(self.conv3_2(
                                  self.relu3_1(self.conv3_1(output_pool2)))))))
        output_pool4 = self.pool4(self.relu4_3(self.conv4_3(
                                  self.relu4_2(self.conv4_2(
                                  self.relu4_1(self.conv4_1(output_pool3)))))))
        output_pool5 = self.pool5(self.relu5_3(self.conv5_3(
                                  self.relu5_2(self.conv5_2(
                                  self.relu5_1(self.conv5_1(output_pool4)))))))
        output_drop6 = self.drop6(self.relu6(self.conv6(output_pool5)))
        output_drop7 = self.drop7(self.relu7(self.fc7(output_drop6)))
        output_score = self.score(output_drop7)

        output_score_deconv1 = self.score_deconv1(output_score)
        output_score_pool4 = self.score_pool4(output_pool4)
        output_score_pool4_cropped = self.crop(output_score_pool4, output_score_deconv1)
        output_score_fused1 = output_score_deconv1 + output_score_pool4_cropped

        output_score_deconv2 = self.score_deconv2(output_score_fused1)
        output_score_pool3 = self.score_pool3(output_pool3)
        output_score_pool3_cropped = self.crop(output_score_pool3, output_score_deconv2)
        output_score_fused2 = output_score_deconv2 + output_score_pool3_cropped

        output_score_deconv3 = self.score_deconv3(output_score_fused2)
        output_score_final = self.crop(output_score_deconv3, batch)

        return output_score_final

    @staticmethod
    def crop(from_mat, to_mat):
        _, _, x1, y1 = from_mat.shape
        _, _, x2, y2 = to_mat.shape
        offset_x, offset_y = int((x1 - x2) / 2), int((y1 - y2) / 2)
        assert offset_x >= 0
        assert offset_y >= 0
        return from_mat[:, :, offset_x:offset_x + x2, offset_y:offset_y + y2]
