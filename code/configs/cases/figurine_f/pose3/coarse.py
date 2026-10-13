_base_ = '../../../custom_e2e/coarse.py'
case_id = 'figurine_f'
pose_id = 3
release_case_config = True
expname = 'figurine_f/pose3'
basedir = './outputs/reconstruction'
WH = [1920, 1080]
expected_image_count = 35
train_all = True
data = dict(datadir='../dataset/figurine_f/pose3', mode=dict(wmask=True))
