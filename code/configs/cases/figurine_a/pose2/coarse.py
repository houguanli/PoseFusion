_base_ = '../../../custom_e2e/coarse.py'
case_id = 'figurine_a'
pose_id = 2
release_case_config = True
expname = 'figurine_a/pose2'
basedir = './outputs/reconstruction'
WH = [800, 800]
expected_image_count = 45
train_all = True
data = dict(datadir='../dataset/figurine_a/pose2', mode=dict(wmask=True))
