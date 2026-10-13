_base_ = '../../../custom_e2e/coarse.py'
case_id = 'heel_slipper'
pose_id = 3
release_case_config = True
expname = 'heel_slipper/pose3'
basedir = './outputs/reconstruction'
WH = [800, 800]
expected_image_count = 40
train_all = True
data = dict(datadir='../dataset/heel_slipper/pose3', mode=dict(wmask=True))
