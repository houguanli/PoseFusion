_base_ = '../../../custom_e2e/fine.py'
case_id = 'heel_slipper'
pose_id = 2
release_case_config = True
expname = 'heel_slipper/pose2'
basedir = './outputs/reconstruction'
WH = [800, 800]
expected_image_count = 40
train_all = False
data = dict(datadir='../dataset/heel_slipper/pose2', mode=dict(wmask=True))
