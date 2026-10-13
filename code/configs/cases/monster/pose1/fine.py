_base_ = '../../../custom_e2e/fine.py'
case_id = 'monster'
pose_id = 1
release_case_config = True
expname = 'monster/pose1'
basedir = './outputs/reconstruction'
WH = [1200, 800]
expected_image_count = 40
train_all = False
data = dict(datadir='../dataset/monster/pose1', mode=dict(wmask=True))
