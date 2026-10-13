_base_ = '../../../custom_e2e/fine.py'
case_id = 'nikon'
pose_id = 1
release_case_config = True
expname = 'nikon/pose1'
basedir = './outputs/reconstruction'
WH = [800, 800]
expected_image_count = 40
train_all = False
data = dict(datadir='../dataset/nikon/pose1', mode=dict(wmask=True))
