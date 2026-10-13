_base_ = '../../../custom_e2e/coarse.py'
case_id = 'nikon'
pose_id = 2
release_case_config = True
expname = 'nikon/pose2'
basedir = './outputs/reconstruction'
WH = [800, 800]
expected_image_count = 40
train_all = True
data = dict(datadir='../dataset/nikon/pose2', mode=dict(wmask=True))
