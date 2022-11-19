import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, Sequential



class BasicBlock(layers.Layer):
    # 殘差模塊
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        # 第一個卷積單元
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 第二個卷積單元
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:# 通過1x1卷積完成shape匹配
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:# shape匹配，直接短接
            self.downsample = lambda x:x

    def call(self, inputs, training=None):

        # [b, h, w, c]，通過第一個卷積單元
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        # 通過第二個卷積單元
        out = self.conv2(out)
        out = self.bn2(out)
        # 通過identity模塊
        identity = self.downsample(inputs)
        # 2條路徑輸出直接相加
        output = layers.add([out, identity])
        output = tf.nn.relu(output) # 激活函數

        return output


class ResNet(keras.Model):
    # 通用的ResNet實現類
    def __init__(self, layer_dims, num_classes=10): # [2, 2, 2, 2]
        super(ResNet, self).__init__()
        # 根網絡，預處理
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
        # 堆疊4個Block，每個block包含了多個BasicBlock,設置步長不一樣
        self.layer1 = self.build_resblock(64,  layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # 通過Pooling層將高寬降低為1x1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 最後連接一個全連接層分類
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        # 通過根網絡
        x = self.stem(inputs)
        # 一次通過4個模塊
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 通過池化層
        x = self.avgpool(x)
        # 通過全連接層
        x = self.fc(x)

        return x



    def build_resblock(self, filter_num, blocks, stride=1):
        # 輔助函數，堆疊filter_num個BasicBlock
        res_blocks = Sequential()
        # 只有第一個BasicBlock的步長可能不為1，實現下采樣
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):#其他BasicBlock步長都為1
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


def resnet18():
    # 通過調整模塊內部BasicBlock的數量和配置實現不同的ResNet
    return ResNet([2, 2, 2, 2])


def resnet34():
     # 通過調整模塊內部BasicBlock的數量和配置實現不同的ResNet
    return ResNet([3, 4, 6, 3])