_base_ = '../../../custom_e2e/coarse.py'
case_id = 'rubik'
pose_id = 1
release_case_config = True
expname = 'rubik/pose1'
basedir = './outputs/reconstruction'
WH = [800, 800]
expected_image_count = 40
train_all = True
data = dict(datadir='../dataset/rubik/pose1', mode=dict(wmask=True))
