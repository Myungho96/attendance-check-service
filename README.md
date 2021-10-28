# attendance-check-service
졸업프로젝트로 진행한 출석 관리 서비스 입니다.

- [x]  프로젝트 소개
- [x]  구현환경 정리
- [x]  설계 : 백엔드 부분
- [x]  구현
    - [x]  모델 만들기
    - [x]  출결 인증
    - [x]  웹
    - [x]  디비
- [x]  결과
- [x]  어려웠던점과 해결
- [x]  느낀점

### 프로젝트 개요

### 1. 프로젝트 소개

대학에서는 보통 두 가지 방식으로 출결을 확인한다. 하나는 교수가 직접 학생의 이름을 호명하는 방식이고, 다른 하나는 출결 번호를 학생이 핸드폰으로 직접 입력하는 방식이다. 그러나 기존의 방식은 수업 시간의 많은 부분을 할애해야 하며, 또한, 일부 학생들의 대리출석이나 ‘출튀(출석만 하고 수업을 듣지 않는 행위)’와 같은 비양심적인 행동으로 부정이 있기도 하다. 이에 따라, 우리는 인공지능 기반으로 학생들의 얼굴을 인식해 출결을 관리하는 프로그램, ‘출결을 부탁해’를 기획하였다.

‘출결을 부탁해’는 강의실의 카메라를 통해 학생의 얼굴을 인식해 지각 여부를 확인하고, 강의실 내부에 얼마나 오래 있었는지에 따라 출결 여부를 결정하는 프로그램이다. 이를 통해, 강의자가 신경 쓰지 않아도 자동으로 학생의 출결 여부를 가리고, 데이터베이스에 저장한다. 이 데이터는 웹상에서 출석부 정보로 불러올 수 있다.

또한, ‘출결을 부탁해’의 핵심 기술인 얼굴 인식 기술은 출결 시스템 뿐만 아니라 다양한 분야에서도 이용될 수 있다. 아직 어려서 보호자의 인솔이 필요한 유치원이나 초등학교의 경우, 교사 1명이 관리하는 학생이 수는 많다. 그렇기 때문에, 교사의 주의에서 멀어지는 학생은 분명 존재할 수밖에 없다. 교사의 시선에서 멀어져, 교실을 벗어난 어린 학생은 사고에 쉽게 노출된다. 이런 상황에서, 학생이 교실에서 사라졌다는 것을 가장 빠르게 알아차리는 것이 사고를 예방하는데 그 무엇보다도 중요하다.

따라서, 다양한 분야에서 이용될 수 있는 ‘인공지능 기반의 얼굴 인식 모델’을 통한 출결 프로그램을 기획하였다.

### 2. 구현 환경

Intel(R) Core(TM) i5-8250U CPU Window10 – Pycharm 2020.3.5. - anaconda - python 3.8

Intel(R) Core(TM) i5-8250U CPU Window10 – Pycharm 2020.1 - anaconda - python 3.7

Intel(R) Core(TM) i7-6700HQ CPU, NVIDIA GeForce GTX 960M GPU Window10 – Pycharm 2021.2.2. - anaconda - python 3.8.12

Google Colab

IntelliJ IDEA 2021.2.3

Flask 2.0.2

java11

Spring Boot 2.5.6

### 3. 프로젝트 설계

프로젝트는 크게 세 가지로 나눌 수 있다. ‘얼굴 인식하는 인공지능 모델 만들기’, ‘모델을 바탕으로 출결 여부 판정하기’, ‘웹/데이터베이스 연동하기’ 이다.

수업을 수강하는 학생 사진을 이용하여 사전에 모델을 구현하고, 이 모델을 이용해 실시간 스트리밍을 진행하도록하였다. 웹에서 총 수업 시간을 입력하고 시작 버튼을 누르면, 웹캠이 시작되며 실시간 스트리밍이 진행된다. 이후, 입력한 시간만큼의 시간이 지나면 수업은 종료되고, 총 출결 여부가 디비에 저장된다. 웹에서 출석부를 확인하면 디비를 확인할 수 있다.

**얼굴 인식하는 인공지능 모델 만들기**

처음부터 모든 모델을 직접 설계하는 것은 성능면에서 한계가 있으므로, ‘transfer learning’을 이용할 예정이다. ‘Transfer learning’은 처음부터 모델을 학습하는 게 아닌, 어느 정도 학습된 패턴을 이용하는 것이다. 만들어진 모델(pre-trained models)을 이용하면 학습에 이용할 데이터의 양도 줄일 수 있으며, 정확도도 올라가고, 동시에 연산도 줄어든다.

이때, 프로젝트에서 선정한 모델은 ‘vggface2’로, 서양인 9131명의 331만장의 사진을 학습시킨 모델이다. ‘Vggface2’를 이용해 학생의 얼굴을 학습하는 과정은 GUP과 장착된 구글 코랩(Google Colab)에서 진행하였으며, 서버에서 완성한 모델을 추출하여 개인 컴퓨터의 파이참에서 완성한 모델을 이용하여 실시간 스트리밍 영상에서 대상의 얼굴을 인식하고, label을 표시하도록 하였다.

**모델을 바탕으로 출결 여부 판정하기**

만들어 놓은 얼굴 인식 모델을 통해 얼굴을 최초로 인식한 시간을 기록하고, 설치된 카메라를 통해서 얼굴을 실시간으로 스트리밍 하여 강의실 내부에 있었던 시간을 출석 여부를 확인한다.

인공지능 모델이 프레임 단위로 이미지를 추출해 얼굴을 인식하는데, 동일한 학생이 같은 자리에 앉아 있어도 어떤 프레임은 본인이라고 인식되는 반면, 어떤 프레임에서는 다른 사람으로 인식하기도 한다. 이런 오차를 보정하기 위해 '지각, 출튀, 출석, 결석'에 대한 기준을 정하였다. 또한, 한 프레임에 한 사람이 여러번 인식될 수 있는데, 이 부분 또한 보정하여 한 프레임에서 한 사람은 한 명만 인식되도록 구현하였다.

수업시간을 입력 받을 수 있도록 매개변수의 형태로 구현하였으며, '지각, 출튀, 출석, 결석' 기준은 입력 받은 수업 시간에 맞춰서 조정될 수 있도록 구현하였다.

**Spring Boot Server 구축**

우리 팀의 목적인 '웹을 이용해 출결 인증 서비스를 제공하기' 를 구현하기 위하여 웹 서버를 구축하였다. 사실 Flask 서버에서도 웹을 구축할 수 있지만, 로드 밸런싱 측면이나 웹의 안정성, 확장성 측면에서 살펴보았을 때 Spring Boot를 통해 구축하기로 하였다.

**기존에 만들었던 출석 인증 프로그램을 플라스크 서버에 이식**

기존에 만들었던 출석 인증 프로그램은 PyCharm에서 값을 넣은 다음 실행하면 사용자가 그만둘 때 까지 출석 체크를 수행하고, 종료되면 정보(출결, 지각, 결석 여부)들을 계산해서 실행하는 구조였다.

이것을 웹에서 사용할 수 있도록 구현하려면 출석 인증 프로그램을 인공지능을 실행할 목적의 서버에 넣어야 api를 통해 실행할 수 있다는 결론에 도달했다.

그래서 pytorch를 원활히 실행 가능한 Flask 서버를 하나 구축했고, 출석 인증 프로그램을 플라스크 서버에 이식해서 api화 하였다.

또한 정해진 시간 동안 실행할 수 있도록 하기 위해서 서버에 동작 시간을 저장할 수 있는 api도 추가하였다.

**데이터베이스 연동하기**

학생들의 출결 여부를 데이터베이스에서 관리할 수 있도록 구현하였다. Mysql DB를 사용했다.

카메라를 통해 영상을 받아 얼굴 인식을 수행하는 플라스크 서버에서는 DB를 연결하고 수업 시간이 끝나면 DB에 학생들의 이름, 전공, 학번, 출석시간, 빈도, 출석결과, 교수님 이름을 저장하도록 구현하였다.  출석 현황을 볼 수 있는 스프링 웹서버에서는 DB에 저장된 데이터를 불러와 화면에 출력할 수 있도록 구현하였다.

**웹과 인공지능 서버를 연동해 실행하기**

웹의 BackEnd를 담당하는 Spring Boot 서버와 출결 인증 프로그램을 작동시키기 위한 Flask 서버가 존재하는데, 우리가 원하는 방식으로 웹이 동작하기 위해서는 두 서버가 연동될 필요가 있었다.

그래서 Flask 서버를 GET 요청을 받을 수 있게 API를 만들었으며, Spring Boot 서버에서 값을 전달하거나 정보를 요청하는 방식으로 웹을 구현하였다.

### 4. 구현

**1) 인공지능 모델**

인공지능 모델 구현은 GPU가 필요하다. 그렇기 때문에 구글 코랩을 이용하였다.

```python
본인의 프로젝트 폴더
    +-- arial.ttf
		+-- make_model.ipynp    #모델 만드는 코드
    +-- data # 폴더
    |   +-- testEmbeds.npz # 코드 실행 중 생성
    |   +-- trainEmbeds.npz # 코드 실행 중 생성
    |   +-- svm.sav # 코드 실행 중 생성
    |   +-- test_images_cropped
        |   +-- person1
            |   +-- 1.png
            |   +-- 2.png
        |   +-- person2
            |   +-- 1.png
            |   +-- 2.png
    |   +-- train_images_cropped
        |   +-- person1
            |   +-- 1.png
            |   +-- 2.png
        |   +-- person2
            |   +-- 1.png
            |   +-- 2.png
		
```

Cropped 이미지를 넣으면, augmentation 작업을 거쳐 데이터의 양을 늘리고, 그 데이터를 embedding 한다. 그 `testEmbeds.npz`와 `trainEmbeds.npz`를 통해서 모델을 만들고 추출하는데, 그게 `svm.sav`이다. 구글 드라이브 내의 폴더는 위와 같이 정리한다.

**Define image path & Preparing data**

```python
# 경로 지정
ABS_PATH = 'drive/My Drive/Project/faceRecog/facenet_code_fix'
DATA_PATH = os.path.join(ABS_PATH, 'data')

# 바로 cropped 된 데이터 폴더의 경로를 ALIGNED_ 에 저장한다.
ALIGNED_TRAIN_DIR = os.path.join(DATA_PATH, 'train_images_cropped')
ALIGNED_TEST_DIR = os.path.join(DATA_PATH, 'test_images_cropped')

# 폴더의 이미지들을 받아와 배열의 형태로 받아오는 함수
def get_files(path='./', ext=('.png', '.jpeg', '.jpg')):
    """ Get all image files """
    files = []
    for e in ext:
        files.extend(glob.glob(f'{path}/**/*{e}'))
    files.sort(key=lambda p: (os.path.dirname(p), int(os.path.basename(p).split('.')[0])))
    return files

# 경로의 이미지를 rgb로 바꾸고, 저장하기
def to_rgb_and_save(path):
    """ Some of the images may have RGBA mode """
    for p in path:
        img = Image.open(p)
        if img.mode != 'RGB':
            img = img.convert('RGB') 
            img.save(p)

# trainF, testF에 cropped 된 것을 넣어준다.
trainF, testF = get_files(ALIGNED_TRAIN_DIR), get_files(ALIGNED_TEST_DIR)

# ALIGNED_TRAIN_DIR, ALIGNED_TEST_DIR 를 받아서 폴더 내부의 데이터 개수를 체크.
trainC, testC = Counter(map(os.path.dirname, trainF)), Counter(map(os.path.dirname, testF))
train_total, train_text  = sum(trainC.values()), '\n'.join([f'\t- {os.path.basename(fp)} - {c}' for fp, c in trainC.items()])
test_total, test_text  = sum(testC.values()), '\n'.join([f'\t- {os.path.basename(fp)} - {c}' for fp, c in testC.items()])

print(f'Train files\n\tpath: {ALIGNED_TRAIN_DIR}\n\ttotal number: {train_total}\n{train_text}')
print(f'Train files\n\tpath: {ALIGNED_TEST_DIR}\n\ttotal number: {test_total}\n{test_text}')

# 이미지를 RGB의 형태로 바꿔야 한다.
to_rgb_and_save(trainF), to_rgb_and_save(testF)
```

경로를 지정하고, 데이터를 처리할 수 있도록 trainF와 testF에 넣고, 학습을 위해 RGB의 형태로 바꿔서 저장한다.

**Plot 함수**

```python
# 폴더 속의 이미지를 plot하는 함수들 : imshow(), plot_gallery(), plot()
# ALIGNED_TRAIN_DIR, ALIGNED_TEST_DIR, augmentation한 데이터를 plot할 때 이용
# augmentation data의 경우는 getEmbeds()를 통해 plot.
def imshow(img, ax, title):  
    ax.imshow(img)
    if title:
        el = Ellipse((2, -1), 0.5, 0.5)
        ax.annotate(title, xy=(1, 0), xycoords='axes fraction', ha='right', va='bottom',
                    bbox=dict(boxstyle="round", fc="0.8"), 
                    arrowprops=dict(arrowstyle="simple", fc="0.6", ec="none", 
                                    patchB=el, connectionstyle="arc3, rad=0.3"))
    ax.set_xticks([]), ax.set_yticks([])

def plot_gallery(images, ncols, nrows, titles=None, title='', figsize=None): 
    if figsize is None: 
        figsize = (18, ncols) if ncols < 10 else (18, 20)  
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.02)

    for i, ax in enumerate(grid): 
        if i == len(images): break 
        imshow(images[i], ax, titles[i] if titles is not None else '')

    y_title_pos = grid[0].get_position().get_points()[1][1] - 0.33 / (1 if nrows == 1 else nrows / 3)
    plt.suptitle(title, y=y_title_pos, fontsize=12)

def plot(paths=None, images=None, titles=None, axtitle=True, title='', to_size=(512, 512)): 
    if paths is not None and len(paths): 
        images = [Image.open(p).resize(to_size) for p in paths]

        nrows = int(ceil(len(images) / 12)) # 12 images per row 
        ncols = 12 if nrows > 1 else len(images)

        if axtitle: 
              titles = [os.path.dirname(p).split('/')[-1] for p in paths]

        plot_gallery(images, ncols, nrows, titles, title)

    elif images is not None and len(images): 
        if isinstance(images, list): 
            images = np.array(images)

        nrows = int(ceil(len(images) / 12)) # 12 images per row 
        ncols = 12 if nrows > 1 else len(images)

        # Rescale
        if images[0].max() > 1: 
            images /= 255. 

        if not isinstance(images, np.ndarray): 
            if images.size(1) == 3 or 1: 
                images = images.permute((0, 2, 3, 1))

        plot_gallery(images, ncols, nrows, titles, title)

    else: 
        raise LookupError('You didnt pass any path or image objects')
    plt.show()
```

폴더 속의 이미지를 plot하는 함수들이다. 모델을 구축하는 것에 직접적으로 이용되지 않지만, 폴더 속의 이미지나 augmentation된 상황을 파악하는데 이용할 수 있다.

다음과 같이 plot을 하면,

```python
plot(paths=trainF, title='Train images')
```

![Untitled](https://user-images.githubusercontent.com/59616266/139295296-d04520a8-d1e7-426a-b3a8-db507fd7c416.png)

이런 결과를 얻을 수 있다.

(사진 부분은 포폴로 사용할 때 편집할 것. 팀원 얼굴 지운 버전으로 올리기.)

**Data transform**

```python
# 이미지를 transform 할 때의 기준(템플릿).
# numpy float 형, 텐서의 형태로, 이미지 정규화하기
standard_transform = transforms.Compose([
                                np.float32, 
                                transforms.ToTensor(),
                                fixed_image_standardization
])

# 그리고 augmentation를 위한 그 템플릿(mask)
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

# 위에서 정의한 standard_transform과 aug_mask를 train과 test data에 적용하도록 한다.
# aug는 train만, standard_transform은 train과 test 둘 모두에 적용한다.
# 아래의 ImageFolder에서 data에 대해 적용한다.
transform = {
    'train': transforms.Compose([
                                 transforms.Lambda(lambd=lambda x: aug_mask(image=np.array(x))['image']),
                                 standard_transform
    ]),
    'test': standard_transform
}
```

모델을 구현하기 위해 기존의 image data를 변형해야 하는데, 그 변형을 위한 transformer이다.

**Data Loader**

```python
b = 32

# train image에 standard_transform만 적용해 trainD 에 넣음.
trainD = datasets.ImageFolder(ALIGNED_TRAIN_DIR, transform=standard_transform)
# Augmented train images. transform을 적용해 trainD_agu에 넣음.
trainD_aug = datasets.ImageFolder(ALIGNED_TRAIN_DIR, transform=transform['train'])
# Train DataLoader
trainL = DataLoader(trainD, batch_size=b, num_workers=2)
trainL_aug = DataLoader(trainD_aug, batch_size=b, num_workers=2)

# test images 
testD = datasets.ImageFolder(ALIGNED_TEST_DIR, transform=standard_transform)
# Test DataLoader
testL = DataLoader(testD, batch_size=b, num_workers=2)

# Convert encoded labels to named claasses
# IDX_TO_CLASS : 사람 이름(index) 배열
IDX_TO_CLASS = np.array(list(trainD.class_to_idx.keys()))
CLASS_TO_IDX = dict(trainD.class_to_idx.items())
```

`Augmentation`을 적용해, 데이터의 수를 늘리고, 그 데이터를 모델을 만드는데 이용하기 위해 정규화 과정을 거친다. 그 후, `Dataloader`로 학습이 가능하도록 데이터를 가져온다.

**Model 선언하기**

```python
from facenet_pytorch import InceptionResnetV1

model = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.5, device=device).eval()
```

선정한 `model`은 `vggface2`라는 서양인 9131명의 331만장의 사진을 학습시킨 모델로, 이 모델을 통해 사람의 얼굴을 감지하는데 이용할 예정이다.

**다량의 Data를 embedding하여 추출하기**

```python
def fixed_denormalize(image): 
    """ Restandartize images to [0, 255]"""
    return image * 128 + 127.5

def getEmbeds(model, n, loader, imshow=False, n_img=5):
    model.eval()
    # image들 저장
    images = []

    embeds, labels = [], []
    # tqdm : 진행바
    for n_i in tqdm.trange(n): 
        for i, (x, y) in enumerate(loader, 1): 
            if imshow and i == 1: 
                inds = np.random.choice(x.size(0), min(x.size(0), n_img))
                images.append(fixed_denormalize(x[inds].data.cpu()).permute((0, 2, 3, 1)).numpy())

            embed = model(x.to(device))
            embed = embed.data.cpu().numpy()
            embeds.append(embed), labels.extend(y.data.cpu().numpy())

    if imshow: 
        # augmentation한 이미지가 담긴 images를 plot.
        plot(images=np.concatenate(images))

    return np.concatenate(embeds), np.array(labels)

# Train data embedding
trainEmbeds, trainLabels = getEmbeds(model, 1, trainL, False)
trainEmbeds_aug, trainLabels_aug = getEmbeds(model, 50, trainL_aug, imshow=True, n_img=3)

trainEmbeds = np.concatenate([trainEmbeds, trainEmbeds_aug])
trainLabels = np.concatenate([trainLabels, trainLabels_aug])

# Test embeddings 
testEmbeds, testLabels = getEmbeds(model, 1, testL, False)

# embedding 한 것들을 '.npz'의 형태로 저장
# 이때, path는 TRAIN_EMBEDS와 TEST_EMBEDS.
TRAIN_EMBEDS = os.path.join(DATA_PATH, 'trainEmbeds.npz')
TEST_EMBEDS = os.path.join(DATA_PATH, 'testEmbeds.npz')

np.savez(TRAIN_EMBEDS, x=trainEmbeds, y=trainLabels)
np.savez(TEST_EMBEDS, x=testEmbeds, y=testLabels)

# 저장한 embedding load
trainEmbeds, trainLabels = np.load(TRAIN_EMBEDS, allow_pickle=True).values()
testEmbeds, testLabels = np.load(TEST_EMBEDS, allow_pickle=True).values()

# 이름 label
trainLabels, testLabels = IDX_TO_CLASS[trainLabels], IDX_TO_CLASS[testLabels]
```

Augmentation 작업을 거친 데이터는 그 양이 많다. 효과적으로 모델을 만들기 위해, 데이터의embedding 작업을 거쳐야 하는데 위의 `getEmbeds()`가 해당 작업을 한다. 후에, train data, aug train data, test data에 대해 `getEmbed()` 함수를 적용하여 데이터(Embeds)와 그때의 target인 Labels를 받아온다.

그 값을 `trainEmbeds.npz`와 `testEmbeds.npz`의 형태로 추출한다. 이 파일은 data 폴더에 저장된다`

**Embeds의 data를 통해 모델 구축하고 추출**

```python
# data
# X : 얼굴 사진 data embedding
# y : taget, 이름 label
X = np.copy(trainEmbeds)
y = np.array([CLASS_TO_IDX[label] for label in trainLabels])

warnings.filterwarnings('ignore', 'Solver terminated early.*')

# 최적의 parameter 찾기 위한 param_grid
param_grid = {'C': [1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 'auto'],
              'kernel': ['rbf', 'sigmoid', 'poly']}
model_params = {'class_weight': 'balanced', 'max_iter': 10, 'probability': True, 'random_state': 3}
model = SVC(**model_params)
clf = GridSearchCV(model, param_grid)
# model 학습
clf.fit(X, y)

# 최적의 parameter와 estimator 출력
print('Best estimator: ', clf.best_estimator_)
print('Best params: ', clf.best_params_)

# 구축된 모델(clf)를 'svm.sav'로 추출
SVM_PATH = os.path.join(DATA_PATH, 'svm.sav')
joblib.dump(clf, SVM_PATH)
clf = joblib.load(SVM_PATH)
```

Embed data와 그 data의 target에 해당하는 사람의 이름 label을 `model`에 넣고 학습을 진행한다. 이때, 학습을 위해 여러가지 `parameter`들을 선정해야 하는데, 이때 어떤 `parameter`가 가장 나은지 알 수 없으므로, `param_grid`과 `model_params`에 후보 `parameter`값을 넣고, 최적의 `parameter`를 자동을 선정하도록 진행하였다.

최종적으로 선정된 모델을 파이참에서 구동하여, 실시간 스트리밍을 진행할 것이므로 모델을 `joblib`를 통해 추출한다. 이때 추출된 모델은 `svm.sav`로 `data` 폴더에 저장된다.

**2) 출결 인증 프로그램**

실시간 스트리밍 영상에서 코랩에서 완성한 모델 svm.sav을 통해 대상을 인식하고, 인식한 것을 바탕으로 출결 여부를 가려야 한다. 다음 진행은 pycharm IDE에서 진행하였다.

```python
본인의 프로젝트 폴더
    +-- arial.ttf
		+-- svm.sav #모델
		+-- attendance_streaming.py  
    +-- templates #폴더/flask 서버
    |   +-- index.html
    +-- data #폴더
    |   +-- test_images_cropped
        |   +-- person1
            |   +-- 1.png
            |   +-- 2.png
        |   +-- person2
            |   +-- 1.png
            |   +-- 2.png
    |   +-- train_images_cropped
        |   +-- person1
            |   +-- 1.png
            |   +-- 2.png
        |   +-- person2
            |   +-- 1.png
            |   +-- 2.png
		
```

파이참 프로젝트에서 다음과 같은 폴더를 구성해야 한다.

**Library**

```python
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import os
import numpy as np
import torch
import joblib
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
import time
from flask import Flask, render_template, Response, request
from flask import stream_with_context

## db 연동을 위한 패키지 추가
import pymysql
```

`OpenCV` : 실시간 이미지 프로세싱 라이브러리. 동영상, 카메라 등을 통해 영상을 전달받아 이미지를 프레임 단위로 가져오기 위해 사용한다.

`facenet_pytorch` : 2015년 구글에서 개발한 얼굴 인식 시스템이다. FaceNet 은 얼굴 인식 시스템의 학습에 사용되는 얼굴에서 특징을 고퀼리티로 추출(face embedding) 할 수 있다.

`torch` : Python을 위한 오픈소스 머신 러닝 라이브러리이다. Torch를 기반으로 하며, 자연어 처리와 같은 애플리케이션을 위해 사용된다.

`PIL` : Python Imaging Library로 파이썬 인터프리터에 이미지 처리와 그래픽 기능을 공유하는 라이브러리이다.

`albumentations` : 데이터를 늘리기 위해 augmentation 작업이 필요하다. 이 작업을 albumentation 라이브러리로 하였다.

`torchvision` : pytorch와 함께 사용되는 computer vision 용 라이브러리로 이미지 및 비디오 변환을 위한 유틸리티이다.

`joblib` : 만든 모델을 저장하고 불러오는 모듈이다. colab 에서 만든 모델을 불러올 때 사용할 라이브러리이다.

`numpy` : 다차원의 배열을 다루는데 특화된 라이브러리이다.

`os` : os 모듈은 운영체제에서 제공되는 여러 기능을 파이썬에서 수행할 수 있게 한다. 파일이나 폴더 관련 부분을 다룰 때 사용한다.

`flask` : 웹에서 실행하기 위한 플라스크 서버.

**People class**

```python
class Person(object):
  def __init__(self,SN,major,name):
    self.SN= SN
    self.major = major
    self.name = name
    self.appear = 0
    self.percent = 0
    self.result = 0

person1 = Person(19930516,'방송연예학','IU')
person2 = Person(22222222,'컴퓨터공학','JAEYOUNG')
person3 = Person(19971124,'컴퓨터공학','JUNE')
person4 = Person(19910923,'시각디자인','KEY')
person5 = Person(19911209, '기계공학', 'MINHO')
person6 = Person(66666666, '컴퓨터공학', 'MYUNGHO')
person7 = Person(19891214, '국문학', 'ONEW')
person8 = Person(19930718, '섬유디자인학', 'TAEMIN')

person_list = [person1, person2, person3, person4, person5, person6, person7, person8]

# anum : streaming에서 각 class 사람들이 인식된 frame의 개수
anum=[ 0, 0, 0, 0, 0, 0, 0, 0]
# 각 사람의 번호
aboard = {0:'IU',1:'JAEYOUNG',2:'JUNE',3:'Key',4:'MINHO',5:'MYUNGHO',6:'ONEW',7:'TAEMIN'}
```

각 사람의 이름과 학번, 학과를 사전에 입력하고 진행한다. 이때, `anum` 은 각 학생의 출석한 정도를 기록할 list이다.

**Draw box**

```python
# 영상에 bounding box를 치고, 대상의 이름을 출력하도록 하는 함수들
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
```

`draw_box()`는 frame에서 추출한 얼굴에 bounding box를 치는 함수이다. 이 함수는 아래의 `preprocess_video()`에서 사용된다.

**Face Extract**

```python
def get_video_embedding(model, x):
    embeds = model(x.to(device))
    return embeds.detach().cpu().numpy()

def face_extract(model, clf, frame, boxes):
    names, prob, idx_list = [], [], []
    if len(boxes):
        x = torch.stack([standard_transform(frame.crop(b)) for b in boxes])
        embeds = get_video_embedding(model, x)
        idx, prob = clf.predict(embeds), clf.predict_proba(embeds).max(axis=1)
        names = [IDX_TO_CLASS[idx_] for idx_ in idx]
        idx_list = list(set(idx.tolist()))
    return names, prob, idx_list
```

`face_extract()`를 통해서, 프레임에서 추출된 사람이 누구인지 파악하고, 이름 index를 `names`에 넣고, 해당 사람일 확률을 `prob`에 넣는다. 이때, output인 `names`와 `prob`는 실시간 스트리밍 영상에서 얼굴 위에 쳐질 bounding box 옆에 같이 뜨게 될 이름과 확률이다. 이때, 각 프레임에서 인식된 사람을 계속해서 count 해야한다. 이를 위해 `idx_list`에 `idx`를 담아서 `names`, `prob`와 함께 return해야한다.

**Live streaming + bounding box**

```python
def preprocess_video(detector, face_extractor, clf, path, absence_per, attend_per, class_time, transform=None, k=3):
    if not transform: transform = lambda x: x.resize((1280, 1280)) if (np.array(x.shape) > 2000).all() else x

    anum = [0, 0, 0, 0, 0, 0, 0, 0]

    framess = 0

    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_class_frame = class_time * fps
    late_num = round(total_class_frame * 0.1)

    # 매 frame마다 진행하도록 loop
    while True:
        ret, frame = capture.read()
        if ret:
            framess = framess + 1

            if framess == 0 or (framess + 1) % k == 0:
                frame = cv2.flip(frame, 1)  # 좌우반전(거울모드)
                iframe = Image.fromarray(transform(frame))

                try:
                    boxes, probs = detector.detect(iframe)
                    if boxes is None: boxes, probs = [], []
                    names, prob, idx = face_extract(face_extractor, clf, iframe, boxes)
                    idx2 = list(set(idx))
                    #################################
                    # anum이 1씩 증가하게 된다.
                    for idx_ in idx2:
                        anum[idx_] = anum[idx_] + 1

                    if (framess + 1) / k == late_num:
                        print("지각 여부 확인")
                        print("When frame is {} : {}".format(late_num, anum))
                        for j in range(len(anum)):
                            # 이때까지 인식이 안 되면, 그 사람의 class의 result를 '지각'으로 설정
                            if anum[j] <= 1:
                                person_list[j].result = '지각(late)'

                    frame_draw = iframe.copy()
                    draw = ImageDraw.Draw(frame_draw)

                    boxes, probs = draw_box(draw, boxes, names, probs)

                except:
                    pass

                frame_np = np.array(frame_draw)
                #cv2.imshow('Face Detection', frame_np)
                rett, buffer = cv2.imencode('.jpg', frame_np)
                frame_np = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_np + b'\r\n')
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if framess >= total_class_frame:
            break

    print(f'Total frames: {framess}')
    ######
    total_time = framess / fps
    framess = framess / k
    atime = [0, 0, 0, 0, 0, 0, 0, 0]           #사람 숫자에 따라 바뀐어야한다.
    # Acheck_TR : Acheck_TR.py 참고하기
    Acheck_TR(anum, atime, person_list, framess, total_time, absence_per, attend_per)
```

매 frame마다 얼굴 인식을 진행하게 되면, 부하가 일어날 수 있으므로 `k` frame마다 다음 처리를 진행하고자 하였다. 이때, 앞선 함수 `face_extract()`를 사용하여 각 프레임에서 인식된 사람의 `name`, `prob`, 그리고 `idx`를 받아온다. 이때, `idx`에 받아온 사람을 `anum`에 1씩 증가하여 출석하였음을 표기한다.

지각 여부는 인식을 하는 도중에 진행해야 하므로, `anum`을 업데이트한 이후에, 지각의 기준이 되는 `late_num`의 숫자보다 작으면 지각으로 처리한다.

프로그램을 실행할 때, 입력한 총 수업 시간 `class_time`을 frame 단위로 바꾼다(`total_class_frame`). 그리고 현재 frame이 `total_class_frame`이 된다면 break하고 loop를 빠져나온다.

이후, 최종 결정된 `anum`에 대해 지각을 제외한 나머지 출결 여부를 결정해야 한다. 이 부분을 위해 `atime` list를 선언하고, `Acheck_TR()` 함수를 진행한다.

**Acheck_TR()**

```python
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
            person_list[i].result = '결석(absence)'

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

        conn.ping()  # mysql 재연결
        # DB에 저장할 데이터 입력
        cursor.execute(sql, (
        person_list[i].SN, person_list[i].major, person_list[i].name, person_list[i].appear, person_list[i].percent, person_list[i].result))
    # 데이터 베이스 반영
    conn.commit()
    # 데이터 베이스 종료
    conn.close()
```

출결의 기준이 될 전체 수업을 참여한 비율에 해당하는 `absence_per`, `attend_per`을 매개변수로 함수에 넘긴다. 이후 `atime` list에 `anum`을 바탕으로 각 학생이 수업에 참여한 총 수업 시간을 넣어준다. 그리고 이 `atime`을 바탕으로 전체 수업 시간에서 총 몇 퍼센트에 해당하는 수업을 참여했는지에 따라 '결석, 출튀, 출석' 여부를 가린다.

그리고 각 class를 디비에 저장한다.

**main**

```python
##### main

if __name__ == '__main__':
    # Define image path
    ABS_PATH = 'C:/Users/dearj/PycharmProjects/datacollecting/210520'
    DATA_PATH = os.path.join(ABS_PATH, 'data')

    # Preparing data
    ALIGNED_TRAIN_DIR = 'C:/Users/dearj/PycharmProjects/datacollecting/210520/data/train_images_cropped'
    ALIGNED_TEST_DIR = 'C:/Users/dearj/PycharmProjects/datacollecting/210520/data/test_images_cropped'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'print Running on device: {device}')

    standard_transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    # augmentation 부분
    # 데이터의 양을 늘리는 부분으로 코랩에서 진행하지만,
    # transform에서 aug_mask가 변수로 들어가기 때문에 남겨두었다.
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

    # batch 사이즈는 32로
    b = 32

    # trainD : train_cropped image를 stadard_transform를 가한 것.
    trainD = datasets.ImageFolder(ALIGNED_TRAIN_DIR, transform=standard_transform)

    # 여기서 폴더에서 뽑아낸 이름이 저장이 된다.
    IDX_TO_CLASS = np.array(list(trainD.class_to_idx.keys()))
    CLASS_TO_IDX = dict(trainD.class_to_idx.items())

    SVM_PATH = os.path.join(ABS_PATH, 'svm_final.sav')
    clf = joblib.load(SVM_PATH)

    inds = range(88)

    #한명이 아닌 여러명이 인식되게 만드는 코드
    standard_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    mtcnn = MTCNN(keep_all=True, min_face_size=70, device=device)
    model = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.6, device=device).eval()

    # 지각의 기준이 될 frame.
    # 지각/출석/출튀/결석의 기준

    # 지각의 기준은 preprocess_video 내부에서 0.1을 기준으로 결정
    # late_num = round(class_time * 0.1)
    # 결석/출석/지각의 퍼센티지 기준이다.
    absence_per = 40
    attend_per = 60

    # live streaming 실행
    print('Processing live stream: ')
    app.run(debug=True)
```

`main()` 부분은 다음과 같다. 자세한 설명과 진행은 코드 내부의 주석으로 적어 두었다.

**3) 웹/디비**

**파이썬 - DB 연동**

```python
## db 연동을 위한 패키지 추가
import pymysql

# db 연동

#host: 접속할 DB 주소, port: RDBMS는 주로 3306 포트를 통해 연결됨, user: DB에 접속할 사용자 ID, passwd: 사용자 비밀번호,
#db: 사용할 DB 이름, charset: 한글이나 유니코드 데이터가 깨지는 것을 막기위한 인코딩 방식 utf8로 설정
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='1234',
    db='test_database',
    charset='utf8'
)

#커서 객체 생성:
#커서는 SQL 문을 실행하거나, 결과를 돌려받는 통로이다.
cursor = conn.cursor()

#연결한 데이터베이스에 새 테이블 생성, 만들어져 있지 않으면 생성하도록 한다.
cursor.execute("""CREATE TABLE IF NOT EXISTS Attendance
               (
               id int PRIMARY KEY not null AUTO_INCREMENT COMMENT '인덱스',
               num varchar(20) not null COMMENT '학번',
               major varchar(20) not null COMMENT '전공',
               name varchar(10) not null COMMENT '이름',
               time int not null COMMENT '출석시간',
               frequency float not null COMMENT '빈도',
               result varchar(20) not null COMMENT '출석',
               day TIMESTAMP not null DEFAULT CURRENT_TIMESTAMP COMMENT '출석일자',
               professor varchar(10) not null DEFAULT '배성일' COMMENT '교수님'
               );"""
                )
# 데이터 입력: 여기서 %s는 일반 문자열 포팅에 사용하는 %d,%ld,%c 등과는 다른 것이다.
# MySQL에서 이것을 Parameter Placeholder라고 하는데 문자열이건 숫자이건 모두 %s를 사용한다.
sql = 'INSERT INTO Attendance(num, major, name, time, frequency, result) VALUES (%s, %s, %s, %s, %s, %s);'

```

파이썬에서 Mysql DB 연동을 위해 `pymysql` 패키지를 추가했다. 사용하고자 하는 DB의 host명, port 번호, 유저명, 패스워드, DB 명, 인코딩 방식을 `pymysql.connect` 메서드를 이용하여 `conn` 변수에 담는다. `conn.cursor()` 메서드로 SQL 문을 실행하거나 결과를 돌려 받을 변수 `cursor`를 생성한다. `cursor.execute` 메서드를 사용하여 DB에 테이블을 생성한다. sql 변수에 DB에 집어넣을 데이터들의 sql 쿼리문을 넣는다.

**연동한 DB에 데이터 저장**

```python
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
            person_list[i].result = '결석(absence)'
            #person_list[i].result = '\033[31m' + '결석(absence)'+ '\033[0m'

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
        #DB에 저장할 데이터 입력 id, num, major, name, time, frequency, result
        cursor.execute(sql, (person_list[i].SN, person_list[i].major, person_list[i].name, person_list[i].appear, person_list[i].percent, person_list[i].result))
    #데이터 베이스 연결
		conn.ping()
		#데이터 베이스 반영
    conn.commit()
    #데이터 베이스 종료
    conn.close()

###############################
```

위에서 작성한 sql 쿼리문의 %s에 인자값을 집어넣어야 DB에 저장이 된다. 앞서 구현한 출석 체크 함수 Acheck_TR 안에서 출석 체크가 다 끝난 후 결과값을 저장한 변수들을 cursor.execute() 메서드로 인자값으로 집어넣는다.

`conn.ping(reconnect = True)` 메서드는 DB가 잘 연결되어 있는지 확인하는 메서드이다. `default` 값으로 `reconnect` 인자가 `True`로 되있어서 생략이 가능하며 DB 연결이 안되어 있다면 기본으로 다시 연결해준다.

`conn.commit()` 메서드를 사용해야 위에서 실행한 쿼리문이 DB에 반영이 된다.

마지막으로 `conn.close()` 메서드를 사용하여 DB를 종료한다.

**Spring Boot - Home 화면 구축 후 구현할 기능 페이지로 이동할 버튼 생성**

```html
<!DOCTYPE HTML>
<html xmlns:th="http://www.thymeleaf.org">
<body>
<div class="container">
    <div>
        <h1>출석을 부탁해</h1>
        <p>출석 메뉴</p>
        <p>
            <a href="/attendanceCheck">수업 시작</a>
            <a href="/attendanceMembers">출석 현황</a>
        </p>
    </div>
</div> <!-- /container -->
</body>
</html>
```

Home 화면으로 이동하면, 우리가 구현할 주요 기능인 수업 시작(시간을 입력하면 서버에 시간을 저장하고, 입력한 시간 동안 출결 체크 시스템을 동작시켜 결과를 DB에 저장) 과 출석 현황(DB를 조회하여 출결 기록을 조회하는 기능)을 수행할 링크로 이동한다.

**시간을 입력하면, 시간을 Flask 서버에 전송하여 저장하는 기능 구현**

```java
@GetMapping("/setTime")
    public String setTime(@RequestParam(value = "time")int time){
        System.out.println(time);
        String testUrl = "http://127.0.0.1:5000/setTime?time="+time;

        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<String> response = restTemplate.getForEntity(testUrl, String.class);
        //assertThat(response.getStatusCode(), equalTo(HttpStatus.OK));
        System.out.println(response.getBody());
        return "attendanceCheck/attendanceCheckVideo";
    }
```

수업 시작 버튼을 누르면 시간을 입력하는 부분이 존재하는데, 시간을 입력한 후 버튼을 누르면 `/setTime`으로 `QueryString`에 `time=?` 을 담아 Controller의 이 코드로 오게 된다.

그러면 시간이 제대로 들어왔는지 콘솔에서 확인한 후, Flask 서버의 `/setTime` 에서 time을 받을 수 있게 url을 생성한다.

그 후, Rest API를 간단하게 이용할 수 있는 `RestTemplate` Class를 생성하여 `testUrl`에 접근하여 response를 `getForEntity`로 받아온다. 성공적으로 `time`을 저장했으면 "저장 성공" 이라는 String이 반환된다.

이것을 콘솔에 출력해 확인한 후, 본격적으로 출결 체크를 진행할 페이지로 이동하게 된다.

**출결 체크 구현**

```html
<!DOCTYPE HTML>
<html xmlns:th="http://www.thymeleaf.org">
<body>
<div class="container">
    <div class="row">
        <div class="col-lg-8  offset-lg-2">
            <h2>출석 체크 중 ...</h2>
            <p>
                <button type="button" onclick="location.href='/' ">홈으로</button>
            </p>
            <iframe src="http://127.0.0.1:5000/" style="display:block; width:100vw; height: 100vh"></iframe>
        </div>
    </div>
</div>
</body>
</html>
```

출결 체크가 실행되는 시간을 앞에서 Flask 서버에 저장한 후 이 부분으로 오게 된다.

이 페이지에서는 iframe을 이용해 웹페이지 안에서 웹페이지를 띄우는 식으로 Flask api를 실행하는데, Flask에서는 요청이 오면 시간이 저장되어 있는지를 체크한 후, 제대로 저장되어 있다면 주어진 시간 동안 출결 체크를 진행해 결과를 DB에 저장하게 구현하였다.

**4) 출석 현황 에서 DB 조회 구현**

**Spring Boot - DB 연동**

```java

	implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
	runtimeOnly 'mysql:mysql-connector-java'
```

build.gradle 파일에서 Mysql 과 jpa 라이브러리를 추가하기위해 dependency를 추가한다.

```java
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
spring.datasource.url=jdbc:mysql://localhost:3306/test_database?serverTimezone=UTC&characterEncoding=UTF-8
spring.datasource.username=root
spring.datasource.password=1234
spring.jpa.show-sql=true
spring.jpa.hibernate.ddl-auto=none
```

사용할 Mysql DB의 정보를 `application.properties`에 추가한다.

`test_database`에는 Mysql에서 사용할 데이터베이스명을 적는다.

`username` 과 `password`에 각각 Mysql에 접속시 필요한 아이디와 비밀번호를 적는다.

`spring.jpa.show-sql`은 쿼리문을 보낼 때 쿼리문을 콘솔에 띄울지 말지 선택하는 옵션이다.

`spring.jpa.hibernate.ddl-auto`은 데이터베이스를 초기화 전략 옵션들이 있다.

- update: 기존의 스키마를 유지하며 JPA에 의해 변경된 부분만 추가한다.
- validate: 엔티티와 테이블이 정상적으로 매핑되어있는지만 검증한다.
- create: 기존에 존재하는 스키마를 삭제하고 새로 생성한다.
- create-drop: 스키마를 생성하고 애플리케이션이 종료될 때 삭제한다.
- none: 초기화 동작을 하지 않는다.

우리 프로젝트의 경우 따로 설정이 필요없으므로 none으로 한다.

**웹서버에서 쓰이는 객체인 Domain**

```java
package hello.hellospring.domain;

import javax.persistence.*;
import java.sql.Timestamp;

//이곳에서 @Table 애노테이션을 통해 DB 테이블명을 바꿔줄 수 있다.
@Entity
@Table(name="attendance")
public class AttendanceMember {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String num;

    private String major;

    private String name;

    private int time;

    private double frequency;

    private String result;

    private Timestamp day;

    private String professor;

    //아래로 getter setter 함수들이 정의되어 있다.생략
}
```

출석부의 인덱스, 학번, 전공, 학생이름, 출석총시간(초), 빈도, 출석결과, 날짜, 교수님 이름 객체를 만들었다. `@Entity` 애노테이션은 이 클래스에서 생성된 객체들을 DB 테이블에 있는 칼럼명들과 매핑시켜준다. DB 테이블 칼럼 순서와 동일하게 객체들을 생성하여 따로 `@Column` 애노테이션으로 일일이 매핑시켜주지 않게 짰다. 클래스 명과 DB 테이블 명이 동일하지 않으므로 `@Table` 애노테이션으로 DB 테이블 명을 명시해주었다.

```java
package hello.hellospring.repository;

import hello.hellospring.domain.AttendanceMember;

import java.util.List;

public interface AttendanceMemberRepository {

    List<AttendanceMember> findAll();
}
```

`@Entity` 애노테이션에 의해 생성된 DB에 접근하는 메서드들을 사용하기 위한 자바 인터페이스 코드를 생성했다. 여기에서 어떤 값을 넣거나, 넣어진 값을 조회하는 등의 CRUD(Create, Read, Update, Delete)를 할 수 있게 된다.

본 프로젝트의 경우 DB에 있는 정보를 그대로 출력하면 되므로 List로 해당 테이블의 모든 칼럼들이 담긴 엔티티를 저장할 `findAll()` 함수만 만들었다.

**자바 클래스 생성**

```java
package hello.hellospring.repository;

import hello.hellospring.domain.AttendanceMember;

import javax.persistence.EntityManager;
import java.util.List;

public class JpaAttendanceMemberRepository implements AttendanceMemberRepository{

    private final EntityManager em;

    public JpaAttendanceMemberRepository(EntityManager em) {
        this.em = em;
    }

    @Override
    public List<AttendanceMember> findAll() {
        List<AttendanceMember> result = em.createQuery("select m from AttendanceMember m", AttendanceMember.class)
                .getResultList();
        return result;
    }
}
```

위에서 만든 자바 인터페이스 클래스 안의 함수들을 정의하는 자바 클래스를 생성했다.

**도메인과 레포지토리를 사용하여 비즈니스 로직을 짜는 서비스 코드를 작성**

```java
package hello.hellospring.service;

import hello.hellospring.domain.AttendanceMember;
import hello.hellospring.repository.AttendanceMemberRepository;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Transactional
public class AttendanceMemberService {

    private final AttendanceMemberRepository attendanceMemberRepository;

    public AttendanceMemberService(AttendanceMemberRepository attendanceMemberRepository){
        this.attendanceMemberRepository = attendanceMemberRepository;
    }

    public List<AttendanceMember> findMembers(){
        return attendanceMemberRepository.findAll();
    }
}
```

**출결현황**

```html
<!DOCTYPE HTML>
<html xmlns:th="http://www.thymeleaf.org">
<body>
<div class="container">
    <div>
        <table>
            <thead>
            <tr>
                <th>#</th>
                <th>학번</th>
                <th>전공</th>
                <th>이름</th>
                <th>출석시간</th>
                <th>출석빈도</th>
                <th>결과</th>
                <th>날짜</th>
                <th>교수님</th>
            </tr>
            </thead>
            <tbody>
            <tr th:each="attendanceMember : ${attendanceMembers}">
                <td th:text="${attendanceMember.id}"></td>
                <td th:text="${attendanceMember.num}"></td>
                <td th:text="${attendanceMember.major}"></td>
                <td th:text="${attendanceMember.name}"></td>
                <td th:text="${attendanceMember.time}"></td>
                <td th:text="${attendanceMember.frequency}"></td>
                <td th:text="${attendanceMember.result}"></td>
                <td th:text="${attendanceMember.day}"></td>
                <td th:text="${attendanceMember.professor}"></td>
            </tr>
            </tbody>
        </table>
        <p>
            <button type="button" onclick="location.href='/' ">돌아가기</button>
        </p>
    </div>
</div> <!-- /container -->
</body>
</html>
```

화면에 보일 출석현황의 attendanceMemberList.html 코드를 짰다.

**Controller 코드**

```java
@GetMapping("/attendanceMembers")
    public String list(Model model){
        List<AttendanceMember> attendanceMembers = attendanceMemberService.findMembers();
        model.addAttribute("attendanceMembers", attendanceMembers);
        return "attendanceMembers/attendanceMemberList";

    }
```

홈페이지 메인 화면에서 출석현황을 누르면 위에서 작성한 html 코드가 보이도록 Controller에 코드를 작성했다.

 

### 5. 결과

![스크린샷(58)](https://user-images.githubusercontent.com/59616266/139295461-a4ebdfb1-8452-45d9-b234-4f5a5415f3d9.png)


localhost:8080 에 접속하면 다음과 같은 화면이 뜬다.

![스크린샷(59)](https://user-images.githubusercontent.com/59616266/139295477-a94ca97d-2a3a-4cc9-a285-03a52d7d70ed.png)


수업 시작 버튼을 누르면, 다음과 같은 화면이 뜬다. 총 수업 시간을 입력하고 실행 버튼을 누르면 실시간 스트리밍 영상이 켜진다.

![7530C830-3E68-4E3F-A78A-42AF02CA8CB5](https://user-images.githubusercontent.com/59616266/139295518-f96f29e5-a600-4ce3-a7de-b8aec5a5f8b9.jpeg)


인식한 대상에 bounding box가 쳐지고, tracking하며 대상을 인식한다. 앞서서 입력한 시간이 지나면 영상은 자동으로 중단되고, 출결 여부는 데이터베이스에 저장된다.

![스크린샷(62)](https://user-images.githubusercontent.com/59616266/139295535-d9a8c6ec-148c-46a0-a3c1-0b37f3bf0410.png)


홈 화면의 출석현황을 누르면 데이터베이스에서 불러온 출석부가 보인다.

### 6. 진행 중 어려웠던 점과 해결

1) 모델의 정확도가 낮으며, 동양인이 서양인에 비해 정확도가 낮다는 문제를 마주했다. 이를 해결하기 위해 여러가지 가설을 세우고 그에 따라 다양한 시도를 하였고, 그 끝에 정확도를 높일 수 있었다.

- UNKNOWN에 해당하는 class를 만든다면 정확도 올라가지 않을까?
    
    → UNKNOWN 클래스를 만들어서 다양한 얼굴을 인식시키고자 하였다. 이 과정에서 AI hub에서 제공하는 K-Face를 제공받아서 다양한 사람들을 UNKNOWN 클래스에 넣은 후 학습을 진행하였다. 이 경우, 얼굴을 좀 더 인식 하는 듯 하였으나, 모든 얼굴을 UNKNOWN 클래스로 인식하는 문제가 발생하였다.
    
- vggface2 모델은 서양인에 대해 학습을 진행한 model이기 때문에 동양인에 대해 정확도가 낮다는 것을 확인할 수 있었다. Pretrained 된 모델을 K-Face data를 이용해 fine tuning을 해보았다.
    
    → fine tuning 작업으로 일부 layer가 아닌 전체 layer의 weight 값이 변하면서, 정확도가 이전보다 저조해졌다.
    
- 데이터의 양과 정확도 사이의 연관성을 분석하였다. 클래스 당 사진을 1-2장만 사용하는 것부터 40장 이상씩 사용한 경우까지 모두 모델을 제작해 정확도를 확인하였다.
    
    → 이미 정교하게 학습된 모델이었기 때문에, 1-2장의 사진으로  weight 값을 미세 조정하는 것이 가장 정확도가 좋았다.
    

2) 웹 구현을 어떻게 할 것인지에 대한 논의가 있었다. 팀원 모두 웹 구현에 익숙치 않았기 때문에 어떻게 설계할 것인지 논의하였다.

- 웹서버부터 인공지능 서버까지 모두 flask 서버로 구현하는 방법과 웹 서버는 spring boot로 구현하고, 인공지능 서버만 flask로 구현하는 방법 두 가지 중에서 서비스 구현 목적과 확장성을 고려하였을 때 후자가 낫다고 판단하여, spring과 flask를 모두 이용하기로 하였다.
- 웹 서버는 Spring Boot 서버가 담당하고, 인공지능 서버는 Flask 서버가 담당하는데, 출결 처리 실시간 스트리밍 영상을 어떻게 front에 띄울지 고민하였다.
    
    기존에 Flask에서는 3프레임마다 얼굴 인식이 실행되며, 프레임에서 얼굴을 인식에 박스와 이름을 적은 후 이미지를 버퍼에 저장하면 버퍼에서 실시간으로 이미지를 뽑아내서 Flask의 프론트에서 보여주는 방법이었다.
    이 과정을 Spring Boot의 프론트에서 띄우려면, flask에서 spring 서버로 이미지를 받아와야하며, 그 이미지를 front에 띄우는 방식으로 구현해야 한다. 하지만 이 과정을 어떻게 구현해야 할지 감이 잡히지 않았다.
    그래서 아예 Flask도 웹서버로서도 동작하게 만들어  Spring Boot에서 시간을 입력하고 실행 버튼을 누르면 새 창으로 Flask가 담당하는 프론트가 실행되도록 구현하였으나, 아쉬움이 있었다.
    이 문제를 해결하기 위해 iframe 기능을 사용하였다. iframe은 웹페이지 내에서 웹페이지를 띄워주는 기능이다. '저장 api'와 '호출 api'를 분리해, '호출 api'를 iframe에 넣어 Spring Boot의 프론트에서 flask의 얼굴 인식 영상을 띄울 수 있었다.
    

### 7. 느낀 점, 소감

인공지능 모델의 정확도 문제 뿐만 아니라, 웹으로 구현하는 것, 이를 위해 서버를 이용하고, 데이터베이스를 연결하는 것까지 수많은 문제를 맞닥뜨렸지만, 우여곡절 끝에 프로젝트가 성공적으로 마무리되었다. 코로나 때문에 프로젝트 진행에도 아쉬운 부분도 있었다. 얼굴 인식을 기반으로 하는 출결 프로그램인데 사람 여러명을 모아서 실험을 진행할 수 없다는 점이다. 이 부분은 사람이 여러명 등장하는 동영상으로 그 성능을 실험하여 보완하였다. 매순간이 불확신의 연속이었지만, 난관을 헤쳐나가는 과정 속에서 개발자로서 문제해결 능력을 길러나갈 수 있었다고 생각한다.

방학 이후에도 금방 끝날 줄 알았던 문제들에 많이 부딪히면서 고민에 고민을 거듭했고, 회의에 회의를 거듭했던 것 같다. 
우리 팀의 주요 기능이 인공지능 구현이었다 보니 여름 방학 때까진 인공지능 구현에 힘썼는데, 팀의 생각처럼 잘 되지 않는 부분들이 많아서 9월 중순까지 인공지능 구현, 개선에 힘을 쓰다 보니 프로젝트 마감 때까지 마감이 가능할까? 하는 생각도 들었고 이걸 웹으로 이식하는 과정, 각 서버를 디비에 연결하는 과정, 두 서버를  연동하는 과정, 다른 서버에서 실행되는 스트리밍 영상을 띄우는 문제 등 많은 난관이 있어왔다고 생각한다. 앞서 졸업 프로젝트를 진행한 선배들이 중간에 벽에 부딪혀서 결국 주제를 바꾸거나 완전 간소화 되는 일들도 많이 봐왔기에 마음을 편히 먹기 더더욱 어려웠다.

그러나 정말 놀랍게도 팀원들이 힘을 합치니 모든 난관을 해쳐나갈 수 있었다. 한 명이 모든 난관을 돌파한 것이 아니라 각자가 난관을 돌아가면서 돌파했다. 만약 혼자서 진행하는 프로젝트 였다면 세 명이 겪은 문제들을 분명히 다 해결하지 못했을 거라고 확신한다. 각자가 한두 발짝 이라도 더 아는 부분이 있다면 해결하려 노력하고, 더 개선하려 노력하고, 적극적으로 회의하여 이뤄내는 과정이 너무 놀라웠고 뿌듯했다.

프로젝트가 끝날 즈음에 돌아보니 코로나라서 아쉬운 부분도 많았던 것 같다. 제일 아쉬운 부분은 프로젝트를 진행하는 내내 코로나가 너무 심해서 사람을 많이 모을 수 없었고, 여러 명이 실제로 웹캠에 있을 때 얼굴이 어느 정도 인식되는가에 대한 문제였다.

이 부분을 보완하기 위해 저장된 동영상 파일을 열어서 얼굴 인식을 할 수 있는 코드를 만들어서 다수가 있을 때 인식 테스트를 진행하여서 괜찮았지만 직접 사람으로 테스트를 많이 하지 못한 부분이 아쉬웠다.
프로젝트 수업도 대면이 아니라 비대면으로 이뤄지다 보니 다른 팀들의 진행 상황이나 서로 필요한 부분에 대한 내용 공유가 안되다 보니 우리 팀 혼자 어두운 동굴을 지나는 기분이었던 것 같다. 물론 더 자유롭게 회의가 가능하고 피드백이 가능하다는 장점도 있었으나 아쉬운 부분들이 보이는 것 같다.


- reference
    pymysql 레퍼런스 링크: [https://pymysql.readthedocs.io/en/latest/modules/index.html#](https://pymysql.readthedocs.io/en/latest/modules/index.html#)
