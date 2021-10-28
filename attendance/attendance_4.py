# face_detection_mp4 : video face detection, 완전 초창기 모델.
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import joblib
import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import shutil
import glob
from PIL import Image, ImageDraw
from PIL import ImageFont
import albumentations as A
from facenet_pytorch import fixed_image_standardization
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import accuracy_score
import time


import imageio
from torch.utils.data import SubsetRandomSampler

# attribute lookup <lambda> on __main__ failed

import sklearn

# attendance
# 사람 관련 정보를 Person class로
# 사람 수에 따라 anum, atime, person_list 변경해야 함
class Person(object):
  def __init__(self,SN,major,name):
    self.SN= SN
    self.major = major
    self.name = name
    self.appear = 0
    self.percent = 0
    self.result = 0

person1 = Person(11111111,'기계공학과','IU')
person2 = Person(22222222,'전자공학과','JAEYOUNG')
person3 = Person(33333333,'전자공학과','JUNE')
person4 = Person(44444444,'화생공학과','KEY')
person5 = Person(55555555, '방송연예학', 'MINHO')
person6 = Person(66666666, '방송연예학', 'MYUNGHO')
person7 = Person(77777777, '방송연예학', 'ONEW')
person8 = Person(88888888, '방송연예학', 'TAEMIN')

person_list = [person1, person2, person3, person4, person5, person6, person7, person8]

# anum : streaming에서 각 class 사람들이 인식된 frame의 개수
anum=[ 0, 0, 0, 0, 0, 0, 0, 0]
# 각 사람의 번호
aboard = {0:'IU',1:'JAEYOUNG',2:'JUNE',3:'Key',4:'MINHO',5:'MYUNGHO',6:'ONEW',7:'TAEMIN'}

###############################
t = time.localtime()

# anum : 각 인물이 잡힌 frame의 개수
# atime : 들어갈 땐 [0,0,0,0]이다
# person_list : person이 답겨있는 list
# frame : 재정의한 frame의 개수
# total_time : 스트리밍된 총 시간
def Acheck_TR(anum, atime, person_list, framess, total_time, absence_per, attend_per):
    for i in range( len( anum ) ) :
        # frame/total_time : 시간 당 frame의 수
        # 인식되었던 시간(frame 단위였던 것을 time으로 변환)
        atime[ i ]=round( anum[ i ] / (framess/total_time) , 1 )
    for i in range( len( atime ) ) :
        # person_list의 appear 항목에 출석한 총 시간 넣어주기
        person_list[ i ].appear=atime[ i ]
    for i in range( len( anum ) ) :
        person_list[ i ].percent=round( anum[i]/framess * 100 , 1 )
        if person_list[i].percent > 100:
          person_list[i].percent = 100
    # 날짜와 시간 출력하기
    s='{}'.format( (lambda x : x if x >= 10 else '0' + '{}'.format( x ))( t.tm_mon ) + '/' ) + '{}'.format(
        (lambda x : x if x >= 10 else '0' + '{}'.format( x ))( t.tm_mday ) ) + ' ' + '{}'.format(
        t.tm_hour ) + ':' + '{}'.format( t.tm_min ) + ':' + '{}'.format( t.tm_sec )
    print('\033[33m'+'[{} 출석체크결과] :'.format( s ))

    # 총 시간, 첫째자리까지 표현하기
    total_time_r = round(total_time,1)
    if total_time >=60 :
        total_time_r = "{}분 {}초".format(total_time_r//60,total_time_r%60)
    else :
        total_time_r = "{}초".format(total_time_r)
    print('\033[96m'+"총 시간: "+total_time_r+ '\033[0m')

    # percent로 출석과 출튀 여부 가리기
    for i in range(len(atime)):
        if person_list[i].percent >= attend_per:
            if person_list[i].result == '지각(late)':
                continue
            else:
                person_list[i].result = '출석(attend)'
        elif person_list[i].percent < attend_per and person_list[i].percent >= absence_per:
            person_list[i].result = '출튀(escape)'
        elif person_list[i].percent < absence_per:
            person_list[i].result = '\033[31m' + '결석(absence)'+ '\033[0m'

    print('결석(absence) : percent < {}'.format(absence_per))
    print('출튀(escape) : {} < percent < {}'.format(absence_per, attend_per))
    print('출석(attend) : {} < percent'.format(attend_per))
    print( '\033[32m'+'|ㅡㅡ학번ㅡㅡ|ㅡㅡㅡ전공ㅡㅡㅡ|ㅡㅡ이름ㅡㅡ|ㅡ출석시간ㅡ|ㅡㅡ빈도ㅡㅡ|ㅡㅡ결과ㅡㅡ|' '\033[0m')
    for i in range( len( person_list ) ) :
        # 총 시간, 첫째자리까지 표현하기
        appear_time_r = round(person_list[i].appear)
        if person_list[i].appear >= 60:
            appear_time_r = "{}분 {}초".format(appear_time_r//60, appear_time_r%60)
        else:
            appear_time_r = "    {}초".format(appear_time_r)

        print('  {}'.format(person_list[i].SN) + '   {}'.format(person_list[i].major) + '    {}'.format(
                person_list[i].name) + '   ' + appear_time_r + '       {}%'.format(
                person_list[i].percent) + '    {}'.format(
                person_list[i].result))
        '''
        print( '  {}'.format( person_list[ i ].SN ) + '   {}'.format( person_list[ i ].major ) + '    {}'.format(
          person_list[ i ].name ) + '      {}초'.format( person_list[ i ].appear ) + '       {}%'.format(
          person_list[ i ].percent )+'    {}'.format(
    person_list[ i ].result ) )
        '''

###############################

# Bounding boxes can overlap, so define some function to avoid duplicates

def diag(x1, y1, x2, y2):
    return np.linalg.norm([x2 - x1, y2 - y1])


def square(x1, y1, x2, y2):
    return abs(x2 - x1) * abs(y2 - y1)


def isOverlap(rect1, rect2):
    x1, x2 = rect1[0], rect1[2]
    y1, y2 = rect1[1], rect1[3]

    x1_, x2_ = rect2[0], rect2[2]
    y1_, y2_ = rect2[1], rect2[3]

    if x1 > x2_ or x2 < x1_: return False
    if y1 > y2_ or y2 < y1_: return False

    rght, lft = x1 < x1_ < x2, x1_ < x1 < x2_
    d1, d2 = 0, diag(x1_, y1_, x2_, y2_)
    threshold = 0.5

    if rght and y1 < y1_:
        d1 = diag(x1_, y1_, x2, y2)
    elif rght and y1 > y1_:
        d1 = diag(x1_, y2_, x2, y1)
    elif lft and y1 < y1_:
        d1 = diag(x2_, y1_, x1, y2)
    elif lft and y1 > y1_:
        d1 = diag(x2_, y2_, x1, y1)

    if d1 / d2 >= threshold and square(x1, y1, x2, y2) < square(x1_, y1_, x2_, y2_): return True
    return False


def draw_box(draw, boxes, names, probs, min_p=0.89):
    font = ImageFont.truetype(os.path.join(ABS_PATH, 'arial.ttf'), size=22)

    not_overlap_inds = []
    for i in range(len(boxes)):
        not_overlap = True
        for box2 in boxes:
            if np.all(boxes[i] == box2): continue
            not_overlap = not isOverlap(boxes[i], box2)
            if not not_overlap: break
        if not_overlap: not_overlap_inds.append(i)

    boxes = [boxes[i] for i in not_overlap_inds]
    probs = [probs[i] for i in not_overlap_inds]
    for box, name, prob in zip(boxes, names, probs):
        if prob >= min_p:
            draw.rectangle(box.tolist(), outline=(255, 255, 255), width=5)
            x1, y1, _, _ = box
            text_width, text_height = font.getsize(f'{name}')
            draw.rectangle(((x1, y1 - text_height), (x1 + text_width, y1)), fill='white')
            draw.text((x1, y1 - text_height), f'{name}: {prob:.2f}', (24, 12, 30), font)

    return boxes, probs


def get_video_embedding(model, x):
    embeds = model(x.to(device))
    return embeds.detach().cpu().numpy()

# model : vggface2
# clf : svm.sav
# frame : 각 frame, iframe에 해당
# boxes : mtcnn.detect를 할 경우 return 되는 값은 boxes : 숫자 4개
def face_extract(model, clf, frame, boxes):
    # names, prob = [], []
    names, prob, idx_list = [], [], []
    # 얼굴이 인식이 되었을 경우(=boxes가 존재할 경우)
    ## att, 사람이 2명이면...?
    if len(boxes):
        x = torch.stack([standard_transform(frame.crop(b)) for b in boxes])
        embeds = get_video_embedding(model, x)
        idx, prob = clf.predict(embeds), clf.predict_proba(embeds).max(axis=1)
        names = [IDX_TO_CLASS[idx_] for idx_ in idx]
        print("names : {}".format(names))
        idx_list = list(set(idx.tolist()))
        print("idx_list : {}".format(idx_list))

    return names, prob, idx_list


# att : 이 부분 왕창 바꿨음
def preprocess_video(detector, face_extractor, clf, path, late_num, absence_per, attend_per, transform=None, k=30):
    frames = []
    if not transform: transform = lambda x: x.resize((1280, 1280)) if (np.array(x.shape) > 2000).all() else x
    capture = cv2.VideoCapture(path)
    start = time.time()
    i = 0
    while True:
        ret, frame = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)

        if not ret: break

        iframe = Image.fromarray(transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if i == 0 or (i + 1) % k == 0:
            boxes, probs = detector.detect(iframe)
            if boxes is None: boxes, probs = [], []
            names, prob, idx = face_extract(face_extractor, clf, iframe, boxes)
            # 한 frame에 여러명 인식되는 것 수정하였음.
            idx2 = list(set(idx))
            #################################
            # anum이 1씩 증가하게 된다.
            for idx_ in idx2:
                anum[idx_] = anum[idx_] + 1

            ## k << 매개변수화할 것
            if (i + 1) / k == late_num:
                print("지각 여부 확인")
                print("When frame is {} : {}".format(late_num,anum))
                for j in range(len(anum)):
                    # 이때까지 인식이 안 되면, 그 사람의 class의 result를 '지각'으로 설정
                    if anum[j] <= 1:
                        person_list[j].result = '지각(late)'

            #################################

        frame_draw = iframe.copy()
        draw = ImageDraw.Draw(frame_draw)

        boxes, probs = draw_box(draw, boxes, names, probs)
        frames.append(frame_draw.resize((620, 480), Image.BILINEAR))
        i += 1

    print(f'Total frames: {i}')
    ######
    total_time = i / fps
    # frame을 재정의하고,
    framess = i / k
    atime = [0, 0, 0, 0, 0, 0, 0, 0]           #사람 숫자에 따라 바뀐어야한다.
    # Acheck_TR : Acheck_TR.py 참고하기
    Acheck_TR(anum, atime, person_list, framess, total_time, absence_per, attend_per)
    return frames


def framesToGif(frames, path):
    with imageio.get_writer(path, mode='I') as writer:
        for frame in tqdm.tqdm(frames):
            writer.append_data(np.array(frame))


print(sklearn.__version__)


if __name__ == '__main__':
    # Define image path
    ABS_PATH = 'C:/Users/myung/PycharmProjects/attendance'
    DATA_PATH = os.path.join(ABS_PATH, 'data')

    # Preparing data
    ALIGNED_TRAIN_DIR = 'C:/Users/myung/PycharmProjects/attendance/data/train_images_cropped'
    ALIGNED_TEST_DIR = 'C:/Users/myung/PycharmProjects/attendance/data/test_images_cropped'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'print Running on device: {device}')

    # Install albumentations for augmentations

    # !pip install albumentations

    # Transformer for data

    standard_transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])



    aug_mask = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.RandomContrast(limit=0.5, p=0.4),
        A.Rotate(30, p=0.2),
        A.RandomSizedCrop((120, 120), 160, 160, p=0.4),
        A.OneOrOther(A.JpegCompression(p=0.2), A.Blur(p=0.2), p=0.66),
        A.OneOf([
            A.Rotate(45, p=0.3),
            A.ElasticTransform(sigma=20, alpha_affine=20, border_mode=0, p=0.2)
        ], p=0.5),
        A.HueSaturationValue(val_shift_limit=10, p=0.3)
    ], p=1)

    transform = {
        'train': transforms.Compose([
            transforms.Lambda(lambd=lambda x: aug_mask(image=np.array(x))['image']),
            standard_transform
        ]),
        'test': standard_transform
    }

    # DataLoader for train/test

    # batch 사이즈는 32로
    b = 32

    # Original train images
    trainD = datasets.ImageFolder(ALIGNED_TRAIN_DIR, transform=standard_transform)



    # Convert encoded labels to named claasses
    IDX_TO_CLASS = np.array(list(trainD.class_to_idx.keys()))
    CLASS_TO_IDX = dict(trainD.class_to_idx.items())

    # Prepare model

    model = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.5, device=device).eval()

    # Get named labels
    # 여기서 폴더에서 뽑아낸 이름이 저장이 된다.
    IDX_TO_CLASS = np.array(list(trainD.class_to_idx.keys()))
    CLASS_TO_IDX = dict(trainD.class_to_idx.items())




    # 학습한 모델 불러오기
    SVM_PATH = os.path.join(ABS_PATH, 'svm_final.sav')
    # joblib.dump(clf, SVM_PATH)
    clf = joblib.load(SVM_PATH)

    # Check the accuracy score on Train & Test datasets(embeddings)

    inds = range(88)
    # Some functional for video preprocessing

    # Create gifs from the existing videos in our path
    VIDEO_PATH = os.path.join(DATA_PATH, 'videos/')
    width, height = 640, 360

    mov1 = os.path.join(VIDEO_PATH, 'test2_av.mp4')
    #toGif(mov1, (width, height))

    #mov2 = os.path.join(VIDEO_PATH, '2.mp4')
    #toGif(mov2, (width, height))

    # Functional for processing video

    standard_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    # Define our models and parameters

    k = 3  # each k image will be processed by networks
    font = ImageFont.truetype(os.path.join(ABS_PATH, 'arial.ttf'), size=22)

    # mtcnn은 얼굴 탐지하는 함수
    mtcnn = MTCNN(keep_all=True, min_face_size=70, device=device)

    # model은 vggface2라는 서양인 9131명의 331만장의 사진을 학습시킨 모델
    # 이 모델을 통해 사람의 얼굴을 감지한다.
    model = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.6, device=device).eval()

    # clf, 우리가 학습시킨 svm.sav 모델

    # 지각의 기준이 될 frame.
    # 매개변수로 만들어둔 지각/출석/출튀/결석의 기준
    late_num = 5
    absence_per = 40
    attend_per = 60

    # Process video and save to gif
    # %%time
    print('Processing mov1: ')
    frames = preprocess_video(mtcnn, model, clf, mov1, late_num, absence_per, attend_per)
    mov1_aug = os.path.join(VIDEO_PATH, 'test2_av_att1.mp4')
    framesToGif(frames, mov1_aug)

