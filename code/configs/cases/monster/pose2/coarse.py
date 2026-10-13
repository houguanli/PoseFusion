_base_ = '../../../custom_e2e/coarse.py'
case_id = 'monster'
pose_id = 2
release_case_config = True
expname = 'monster/pose2'
basedir = './outputs/reconstruction'
WH = [1200, 800]
expected_image_count = 40
train_all = True
data = dict(datadir='../dataset/monster/pose2', mode=dict(wmask=True))
