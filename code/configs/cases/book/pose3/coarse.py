_base_ = '../../../custom_e2e/coarse.py'
case_id = 'book'
pose_id = 3
release_case_config = True
expname = 'book/pose3'
basedir = './outputs/reconstruction'
WH = [800, 800]
expected_image_count = 45
train_all = True
data = dict(datadir='../dataset/book/pose3', mode=dict(wmask=True))
