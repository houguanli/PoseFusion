_base_ = '../../../custom_e2e/fine.py'
case_id = 'guardian'
pose_id = 3
release_case_config = True
expname = 'guardian/pose3'
basedir = './outputs/reconstruction'
WH = [1024, 1024]
expected_image_count = 60
train_all = False
data = dict(datadir='../dataset/guardian/pose3', mode=dict(wmask=True))
