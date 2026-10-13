_base_ = '../../../custom_e2e/fine.py'
case_id = 'book'
pose_id = 1
release_case_config = True
expname = 'book/pose1'
basedir = './outputs/reconstruction'
WH = [800, 800]
expected_image_count = 45
train_all = False
data = dict(datadir='../dataset/book/pose1', mode=dict(wmask=True))
