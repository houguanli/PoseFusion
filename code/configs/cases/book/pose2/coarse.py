_base_ = '../../../custom_e2e/coarse.py'
case_id = 'book'
pose_id = 2
release_case_config = True
expname = 'book/pose2'
basedir = './outputs/reconstruction'
WH = [800, 800]
expected_image_count = 45
train_all = True
data = dict(datadir='../dataset/book/pose2', mode=dict(wmask=True))
