_base_ = '../../../custom_e2e/coarse.py'
case_id = 'lego'
pose_id = 2
release_case_config = True
expname = 'lego/pose2'
basedir = './outputs/reconstruction'
WH = [800, 800]
expected_image_count = 40
train_all = True
data = dict(datadir='../dataset/lego/pose2', mode=dict(wmask=True))
