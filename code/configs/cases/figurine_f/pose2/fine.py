_base_ = '../../../custom_e2e/fine.py'
case_id = 'figurine_f'
pose_id = 2
release_case_config = True
expname = 'figurine_f/pose2'
basedir = './outputs/reconstruction'
WH = [1920, 1080]
expected_image_count = 36
train_all = False
data = dict(datadir='../dataset/figurine_f/pose2', mode=dict(wmask=True))
