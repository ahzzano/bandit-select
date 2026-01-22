import openvino.runtime as ov
from bandit import OVBandit
from solvers import RoundRobin
import cv2
import numpy as np
import os
import random
import argparse

parser = argparse.ArgumentParser(prog='BanditProblem')
parser.add_argument('model')
parser.add_argument('images')
args = parser.parse_args()

def get_images(_dir):
    classes = os.listdir(_dir)
    img_files = []
    for i in classes:
        class_dir = f'{_dir}/{i}/'
        files = os.listdir(class_dir)
        for f in files:
            file_name = f'{class_dir}{f}'
            img_files.append(file_name)

    random.shuffle(img_files)
    # print(img_files)

    for file_name in img_files:
        img = cv2.imread(file_name)
        img = cv2.resize(img, (224,224))
        img = np.expand_dims(img, axis=0)
        yield file_name, img

def main():
    core = ov.Core()
    model = core.read_model(model=args.model)

    model.reshape([1,224,224,3])
    port = model.input(0).shape
    print(port)

    bandit = OVBandit([
        core.compile_model(model=model, device_name="CPU"),
        core.compile_model(model=model, device_name="MYRIAD")
       ], RoundRobin(2))

    for N in range(10):
        selected = 0
        total_time = 0
        for name, i in get_images(args.images):
            bandit.choose(selected)
            time, output = bandit.predict(i)
            selected = (selected + 1) % 2
            total_time += time
            # print(f"{time}ms")
        total_time_secs = total_time/1000
        print(f"[RoundRobin] R#{N} {total_time}ms {total_time_secs}s")

        total_time = 0
        bandit.choose(0)
        for name, i in get_images(args.images):
            time, output = bandit.predict(i)
            selected = (selected + 1) % 2
            total_time += time
        total_time_secs = total_time/1000
        print(f"[CPU] R#{N} {total_time}ms {total_time_secs}s")
        total_time = 0
        bandit.choose(1)
        for name, i in get_images(args.images):
            time, output = bandit.predict(i)
            selected = (selected + 1) % 2
            total_time += time
        total_time_secs = total_time/1000
        print(f"[NCS] R#{N} {total_time}ms {total_time_secs}s")


if __name__ == '__main__':
    main()