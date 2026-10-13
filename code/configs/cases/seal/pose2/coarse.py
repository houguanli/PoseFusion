_base_ = '../../../custom_e2e/coarse.py'
case_id = 'seal'
pose_id = 2
release_case_config = True
expname = 'seal/pose2'
basedir = './outputs/reconstruction'
WH = [800, 800]
expected_image_count = 40
train_all = True
data = dict(datadir='../dataset/seal/pose2', mode=dict(wmask=True))
